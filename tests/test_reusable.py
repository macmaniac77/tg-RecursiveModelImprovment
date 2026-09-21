import importlib.util
import math
import json
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('tinygrad')
from tinygrad import Tensor
from tinygrad.helpers import Context
from tinygrad.nn.state import get_state_dict
from tgaoi import aoi
from tgaoi.audit import catalog, audit_graph
from tgaoi.backends import TinygradBackend
from tgaoi.compose import inline
from tgaoi.ir import Graph, TensorType


def run(g, feeds):
    g = Graph.from_dict(json.loads(json.dumps(g.to_dict())))
    return TinygradBackend().run(g, feeds)[g.outputs[0]].numpy()


@pytest.mark.parametrize('kind', ['relu', 'silu', 'gelu', 'sigmoid', 'tanh'])
def test_activations(kind):
    x = np.array([-10, -2, -0.1, 0, 0.3, 2, 10], dtype=np.float32)
    sig = 1/(1+np.exp(-x))
    expected = {'relu':np.maximum(x,0), 'silu':x*sig, 'gelu':x*0.5*(1+np.array([math.erf(float(v)/math.sqrt(2)) for v in x])),
                'sigmoid':sig, 'tanh':np.tanh(x)}[kind]
    np.testing.assert_allclose(run(aoi.activation_graph(kind), {'x': x}), expected, atol=2e-6, rtol=2e-5)


def test_layernorm_and_attention():
    rng=np.random.default_rng(10)
    x=rng.normal(size=(2,3,4)).astype('float32');w=rng.normal(size=4).astype('float32');b=w/2
    expected=(x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)*w+b
    np.testing.assert_allclose(run(aoi.layernorm_graph(), {'x':x,'weight':w,'bias':b,'eps':1e-5}),expected,atol=2e-6)
    q=rng.normal(size=(1,2,3,4)).astype('float32');k=q*0.8;v=q-0.5
    scores=q@k.swapaxes(-1,-2)*0.5;ex=np.exp(scores-scores.max(-1,keepdims=True));expected=(ex/ex.sum(-1,keepdims=True))@v
    np.testing.assert_allclose(run(aoi.attention_graph(),{'q':q,'k':k,'v':v,'scale':0.5}),expected,atol=2e-6)


def test_mlp_swiglu_and_residual():
    rng=np.random.default_rng(9);x=rng.normal(size=(1,2,3)).astype('float32')
    for g in [aoi.mlp_graph(3,5,3,'relu'),aoi.swiglu_graph(3,5,3)]:
        feeds={k:rng.normal(size=t.shape).astype('float32') for k,t in g.inputs.items() if k!='x'};feeds['x']=x
        if g.name=='DENSE_MLP':
            expected=np.maximum(x@feeds['fc1.weight'].T+feeds['fc1.bias'],0)@feeds['fc2.weight'].T+feeds['fc2.bias']
        else:
            gate=x@feeds['gate.weight'].T
            expected=((gate/(1+np.exp(-gate)))*(x@feeds['up.weight'].T))@feeds['down.weight'].T
        np.testing.assert_allclose(run(g,feeds),expected,atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(run(aoi.residual_graph(),{'x':x,'branch':x}),2*x,atol=1e-6)


def test_pooling():
    x=np.arange(16,dtype='float32').reshape(1,1,4,4)
    for kind,expected in [('max',[[5,7],[13,15]]),('avg',[[2.5,4.5],[10.5,12.5]])]:
        np.testing.assert_allclose(run(aoi.pool2d_graph(kind),{'x':x}),np.array(expected).reshape(1,1,2,2))


spec=importlib.util.spec_from_file_location('yolo_fixture',Path(__file__).parent/'fixtures/yolov8_blocks.py')
yolo=importlib.util.module_from_spec(spec);spec.loader.exec_module(yolo)


@pytest.mark.parametrize('kind,config', [
    ('conv', {'c1':4,'c2':4,'kernel_size':3,'stride':2,'groups':2,'dilation':2}),
    ('bottleneck', {'c1':4,'c2':4,'shortcut':True}),
    ('bottleneck', {'c1':4,'c2':6,'shortcut':False}),
    ('c2f', {'c1':4,'c2':4,'repeats':0}),
    ('c2f', {'c1':4,'c2':4,'repeats':2,'shortcut':True}),
    ('sppf', {'c1':4,'c2':4,'kernel_size':3}),
])
def test_yolo_blocks_match_pinned_upstream(kind,config):
    graph=getattr(aoi,'yolov8_'+kind+'_graph')(**config)
    args=dict(config)
    if kind=='conv': model=yolo.Conv_Block(**args)
    elif kind=='bottleneck': model=yolo.Bottleneck(**args)
    elif kind=='c2f':
        args['n']=args.pop('repeats');model=yolo.C2f(**args)
    else:
        args['k']=args.pop('kernel_size');model=yolo.SPPF(**args)
    rng=np.random.default_rng(8);state=get_state_dict(model);feeds={}
    for key,typ in graph.inputs.items():
        if key=='x':continue
        source_key=key.replace('bn.mean','bn.running_mean').replace('bn.variance','bn.running_var')
        for i in range(config.get('repeats',0)):source_key=source_key.replace(f'block{i}.',f'bottleneck.{i}.')
        assert tuple(state[source_key].shape)==tuple(typ.shape)
        values=rng.normal(0,0.3,size=typ.shape).astype('float32')
        if key.endswith('variance'):values=np.abs(values)+0.5
        state[source_key].assign(Tensor(values)).realize();feeds[key]=values
    feeds['x']=rng.normal(size=(1,config['c1'],5,5)).astype('float32')
    with Context(TRAINING=0):expected=model(Tensor(feeds['x'])).numpy()
    np.testing.assert_allclose(run(graph,feeds),expected,atol=3e-5,rtol=3e-5)


def test_all_public_builders_and_composition_contract():
    for name in aoi.__all__:getattr(aoi,name)().validate()
    for name in ['YOLOV8_C2F','YOLOV8_SPPF','DENSE_MLP','SWIGLU','ATTENTION']:
        assert audit_graph(catalog()[name])['status']=='lowerable'
    parent=Graph('parent',{'x':TensorType((2,))},[])
    with pytest.raises(ValueError,match='exact input'):inline(parent,aoi.linear_graph(),'bad',{'x':'x'})
    inline(parent,aoi.activation_graph(),'relu',{'x':'x'})
    with pytest.raises(ValueError,match='duplicate'):inline(parent,aoi.activation_graph(),'relu',{'x':'x'})
