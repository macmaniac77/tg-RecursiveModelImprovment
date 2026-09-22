import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('tinygrad')
from tinygrad import Tensor
from tgaoi import aoi
from tgaoi.ir import Graph
from tgaoi.backends import TinygradBackend

spec=importlib.util.spec_from_file_location('qwen_reference',Path(__file__).parent/'fixtures/qwen_math.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)


def run(g, feeds):
    graph=Graph.from_dict(json.loads(json.dumps(g.to_dict())))
    return {k:v.numpy() for k,v in TinygradBackend().run(graph,feeds).items()}


def rand(shape, seed=3):
    return np.random.default_rng(seed).normal(0,0.3,size=shape).astype('float32')


@pytest.mark.parametrize('head_dim,rope_dim',[(8,4),(8,8)])
def test_rotary_against_upstream(head_dim,rope_dim):
    freq=run(aoi.rotary_frequencies_graph(rope_dim,3),{'positions':np.arange(5,8,dtype='float32')})
    upstream_freq=reference.precompute_freqs_cis(rope_dim,8).numpy()[5:8]
    np.testing.assert_allclose(np.concatenate([freq['cos'],freq['sin']],-1),upstream_freq,atol=2e-6)
    x=rand((2,3,3,head_dim))
    expected=reference.apply_rope(Tensor(x[...,:rope_dim].copy()),Tensor(upstream_freq)).numpy()
    expected=np.concatenate([expected,x[...,rope_dim:]],-1)
    actual=run(aoi.rope_graph(head_dim,rope_dim),{'x':x,**freq})['out']
    np.testing.assert_allclose(actual,expected,atol=2e-6)


def test_l2_zero_and_small_inputs():
    x=np.array([[0,0],[1e-8,0],[3,4]],dtype='float32')
    expected=Tensor(x).normalize(dim=-1,eps=1e-6).numpy()
    np.testing.assert_allclose(run(aoi.l2_normalize_graph(),{'x':x})['out'],expected,atol=1e-6)


@pytest.mark.parametrize('tokens,prefix',[(3,0),(2,3),(1,4)])
def test_gqa_cached_causal_mask(tokens,prefix):
    q=rand((1,4,tokens,4));k=rand((1,2,tokens+prefix,4),4);v=rand(k.shape,5)
    mask=np.where(np.arange(tokens+prefix)[None,:]<=prefix+np.arange(tokens)[:,None],0.,-np.inf).astype('float32')[None,None]
    expected=Tensor(q).scaled_dot_product_attention(Tensor(k),Tensor(v),attn_mask=Tensor(mask),enable_gqa=True).numpy()
    actual=run(aoi.grouped_query_attention_graph(),{'q':q,'k':k,'v':v,'mask':mask})['out']
    np.testing.assert_allclose(actual,expected,atol=2e-6)


def test_kv_append_preserves_inputs():
    feeds={'past_k':rand((1,2,2,4)),'past_v':rand((1,2,2,4),4),'k':rand((1,2,1,4),5),'v':rand((1,2,1,4),6)}
    copies={k:v.copy() for k,v in feeds.items()};result=run(aoi.kv_append_graph(),feeds)
    for v in ('k','v'):np.testing.assert_array_equal(result['next_'+v],np.concatenate([feeds['past_'+v],feeds[v]],2))
    for k in feeds:np.testing.assert_array_equal(feeds[k],copies[k])


def test_convolution_streaming_state():
    x=rand((1,4,3));state=rand((1,2,3),4);weight=rand((3,3),5)
    result=run(aoi.causal_depthwise_conv_graph(3,3,4),{'x':x,'state':state,'weight':weight})
    window=np.concatenate([state,x],1);expected=sum(window[:,i:i+4]*weight[:,i] for i in range(3))
    expected=expected/(1+np.exp(-expected))
    np.testing.assert_allclose(result['out'],expected,atol=2e-6)
    first=run(aoi.causal_depthwise_conv_graph(3,3,2),{'x':x[:,:2].copy(),'state':state,'weight':weight})
    second=run(aoi.causal_depthwise_conv_graph(3,3,2),{'x':x[:,2:].copy(),'state':first['next_state'],'weight':weight})
    np.testing.assert_allclose(np.concatenate([first['out'],second['out']],1),result['out'],atol=2e-6)
    np.testing.assert_array_equal(second['next_state'],result['next_state'])


def test_delta_gates_and_gated_norm():
    logits=np.array([[[-100.,0.,100.]]],dtype='float32');dt=np.array([0.1,0.2,-0.1],dtype='float32');a=np.array([-1.,-2.,-0.1],dtype='float32')
    result=run(aoi.delta_gates_graph(),{'alpha_logits':logits,'beta_logits':logits,'dt_bias':dt,'ssm_a':a})
    np.testing.assert_allclose(result['decay'],np.exp(np.logaddexp(0,logits+dt)*a),atol=2e-6)
    np.testing.assert_allclose(result['beta'],Tensor(logits).sigmoid().numpy(),atol=2e-6)
    x=rand((1,2,4));gate=rand(x.shape,5);w=rand((4,),6)
    expected=x/np.sqrt((x*x).mean(-1,keepdims=True)+1e-6)*w*(gate/(1+np.exp(-gate)))
    np.testing.assert_allclose(run(aoi.gated_rmsnorm_graph(),{'x':x,'weight':w,'gate':gate})['out'],expected,atol=2e-6)


def test_delta_scan_source_equivalence_and_chunk_continuity():
    feeds={'q':rand((1,2,3,4)),'k':rand((1,2,3,4),4),'v':rand((1,2,3,3),5),
           'beta':np.full((1,2,3),0.6,dtype='float32'),'decay':np.full((1,2,3,1),0.8,dtype='float32'),'state':rand((1,2,3,4),6)}
    expected,state=reference.reference_delta(*[Tensor(feeds[k]) for k in ('q','k','v','beta','decay','state')])
    result=run(aoi.gated_delta_scan_graph(),feeds)
    np.testing.assert_allclose(result['out'],expected.numpy(),atol=2e-6)
    np.testing.assert_allclose(result['next_state'],state.numpy(),atol=2e-6)
    carried=feeds['state'];outputs=[]
    for i in range(3):
        chunk={k:v[:,:,i:i+1].copy() for k,v in feeds.items() if k!='state'};chunk['state']=carried
        step=run(aoi.gated_delta_scan_graph(tokens=1),chunk);carried=step['next_state'];outputs.append(step['out'])
    np.testing.assert_allclose(np.concatenate(outputs,2),result['out'],atol=2e-6)
    np.testing.assert_allclose(carried,result['next_state'],atol=2e-6)
    feeds['beta'].fill(0);feeds['decay'].fill(1)
    np.testing.assert_array_equal(run(aoi.gated_delta_scan_graph(),feeds)['next_state'],feeds['state'])


@pytest.mark.parametrize('builder,kwargs',[(aoi.rope_graph,{'rope_dim':3}),(aoi.grouped_query_attention_graph,{'query_heads':3}),
                                         (aoi.gated_delta_scan_graph,{'tokens':0}),(aoi.causal_depthwise_conv_graph,{'kernel':1})])
def test_invalid_configs_rejected(builder,kwargs):
    with pytest.raises(ValueError):builder(**kwargs)


def test_actual_uop_inspection_is_topological_and_omits_values():
    g=aoi.gated_delta_step_graph()
    feeds={'q':rand((1,2,1,4)), 'k':rand((1,2,1,4)), 'v':rand((1,2,3,1)),
           'beta':np.ones((1,2,1,1),dtype='float32'), 'decay':np.ones((1,2,1,1),dtype='float32'),
           'state':rand((1,2,3,4))}
    trace=TinygradBackend().inspect_uops(g,feeds)
    assert trace['stage']=='lazy_tensor_uops_before_scheduling'
    assert set(trace['outputs'])=={'out','next_state'}
    assert any(n['op']=='MUL' for n in trace['nodes'])
    for node in trace['nodes']:
        assert all(i<node['id'] for i in node['inputs'])
        assert set(node)=={'id','op','dtype','inputs'}
    json.dumps(trace)
