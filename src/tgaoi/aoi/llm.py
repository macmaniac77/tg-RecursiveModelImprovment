"""Pure graph building blocks from Tinygrad's Qwen-capable LLM path.

Reference: tinygrad/llm/model.py at b7ff6dedf7e385b603bf9442ee23c6adf6601fb4.
State updates are explicit returned values, not hidden UOp STORE/AFTER effects.
These float32 reference graphs do not emulate quantization or fused GPU kernels.
"""
import math
from ..ir import Graph, Node, TensorType
from ..compose import inline
from .math import softmax_graph
from .nn import rmsnorm_graph
from .vision import activation_graph


def _positive(**dims):
    if any(type(v) is not int or v <= 0 for v in dims.values()):
        raise ValueError(f'positive integer dimensions required: {dims}')


def _graph(name, inputs, outputs, source):
    return Graph(name, {k: TensorType(tuple(v)) for k,v in inputs.items()}, outputs,
                 metadata={'source': 'tinygrad/llm/model.py::'+source,
                           'precision': 'float32 reference', 'state': 'explicit inputs/outputs'})


def _slice(g, name, value, axis, start, stop):
    return g.add(Node(name, 'SLICE', [value], {'axis':axis,'start':start,'stop':stop}))


def _reshape(g, name, value, shape):
    return g.add(Node(name,'RESHAPE',[value],{'shape':list(shape)}))


def _const(g, name, value):
    return g.add(Node(name,'CONST',attrs={'value':value}))


def rotary_frequencies_graph(dim=4, tokens=3, theta=10000.0):
    _positive(dim=dim,tokens=tokens)
    if dim % 2 or not math.isfinite(theta) or theta <= 0:
        raise ValueError('even rotary dimension and finite positive theta required')
    g=_graph('ROTARY_FREQUENCIES',{'positions':(tokens,)},['cos','sin'],'precompute_freqs_cis')
    inv=_const(g,'inverse_frequencies',[[theta**(-i/dim) for i in range(0,dim,2)]])
    pos=_reshape(g,'positions_column','positions',(tokens,1))
    angle=g.add(Node('angles','MUL',[pos,inv]))
    g.add(Node('cos','COS',[angle]));g.add(Node('sin','SIN',[angle]))
    g.validate();return g


def rope_graph(head_dim=8, rope_dim=4):
    _positive(head_dim=head_dim,rope_dim=rope_dim)
    if rope_dim % 2 or rope_dim > head_dim:
        raise ValueError('rope_dim must be even and <= head_dim')
    g=_graph('ROPE_HALF_SPLIT',{'x':('B','H','T',head_dim),'cos':('T',rope_dim//2),
              'sin':('T',rope_dim//2)},['out'],'apply_rope')
    a=_slice(g,'first','x',-1,0,rope_dim//2);b=_slice(g,'second','x',-1,rope_dim//2,rope_dim)
    ac=g.add(Node('ac','MUL',[a,'cos']));bs=g.add(Node('bs','MUL',[b,'sin']))
    bc=g.add(Node('bc','MUL',[b,'cos']));ass=g.add(Node('as','MUL',[a,'sin']))
    left=g.add(Node('left','SUB',[ac,bs]));right=g.add(Node('right','ADD',[bc,ass]))
    parts=[left,right]
    if rope_dim<head_dim:parts.append(_slice(g,'tail','x',-1,rope_dim,head_dim))
    g.add(Node('out','CONCAT',parts,{'axis':-1}));g.validate();return g


def l2_normalize_graph(eps=1e-6):
    if not math.isfinite(eps) or eps<=0:raise ValueError('positive finite epsilon required')
    g=_graph('L2_NORMALIZE',{'x':('...','D')},['out'],'GatedDeltaNetBlock._attention')
    sq=g.add(Node('square','MUL',['x','x']))
    total=g.add(Node('sum','REDUCE_SUM',[sq],{'axis':-1,'keepdim':True}))
    norm=g.add(Node('norm','SQRT',[total]));e=_const(g,'epsilon',eps)
    den=g.add(Node('den','MAXIMUM',[norm,e]))
    g.add(Node('out','DIV',['x',den]));g.validate();return g


def grouped_query_attention_graph(query_heads=4, kv_heads=2, head_dim=4):
    _positive(query_heads=query_heads,kv_heads=kv_heads,head_dim=head_dim)
    if query_heads%kv_heads:raise ValueError('query_heads must be divisible by kv_heads')
    g=_graph('MASKED_GQA',{'q':('B',query_heads,'T',head_dim),'k':('B',kv_heads,'S',head_dim),
              'v':('B',kv_heads,'S',head_dim),'mask':(1,1,'T','S')},['out'],'TransformerBlock._attention')
    g.metadata['mask']='Explicit additive mask. Caller uses lower-right causal alignment for cached decoding; no entirely masked rows.'
    expanded=[]
    for value in ('k','v'):
        parts=[]
        for h in range(kv_heads):
            part=_slice(g,f'{value}{h}',value,1,h,h+1)
            parts += [part]*(query_heads//kv_heads)
        expanded.append(g.add(Node(value+'_heads','CONCAT',parts,{'axis':1})))
    kt=g.add(Node('kt','TRANSPOSE',[expanded[0]],{'axes':[-1,-2]}))
    scores=g.add(Node('scores','MATMUL',['q',kt]));scale=_const(g,'scale',head_dim**-0.5)
    scores=g.add(Node('scaled','MUL',[scores,scale]));scores=g.add(Node('masked','ADD',[scores,'mask']))
    weights=inline(g,softmax_graph(),'softmax',{'x':scores})[0]
    g.add(Node('out','MATMUL',[weights,expanded[1]]));g.validate();return g


def kv_append_graph():
    g=_graph('KV_APPEND',{'past_k':('B','H','P','D'),'past_v':('B','H','P','D'),
              'k':('B','H','T','D'),'v':('B','H','T','D')},['next_k','next_v'],'TransformerBlock._attention')
    for value in ('k','v'):g.add(Node('next_'+value,'CONCAT',['past_'+value,value],{'axis':2}))
    g.metadata['scope']='Append-only valid prefix. Caller handles reset, capacity, position and cache dtype; no in-place store.'
    g.validate();return g


def causal_depthwise_conv_graph(channels=4, kernel=3, tokens=2):
    _positive(channels=channels,kernel=kernel,tokens=tokens)
    if kernel<2:raise ValueError('stateful reference requires kernel >= 2')
    g=_graph('CAUSAL_DEPTHWISE_CONV_STATE',{'x':('B',tokens,channels),'state':('B',kernel-1,channels),
              'weight':(channels,kernel)},['out','next_state'],'GatedDeltaNetBlock._attention')
    window=g.add(Node('window','CONCAT',['state','x'],{'axis':1}));parts=[]
    for i in range(kernel):
        rows=_slice(g,f'rows{i}',window,1,i,i+tokens)
        w=_slice(g,f'w{i}','weight',1,i,i+1);w=_reshape(g,f'weight{i}',w,(1,1,channels))
        parts.append(g.add(Node(f'weighted{i}','MUL',[rows,w])))
    total=parts[0]
    for i,p in enumerate(parts[1:]):total=g.add(Node(f'sum{i}','ADD',[total,p]))
    out=inline(g,activation_graph('silu'),'silu',{'x':total})[0]
    g.add(Node('out','IDENTITY',[out]));_slice(g,'next_state',window,1,tokens,tokens+kernel-1)
    g.validate();return g


def delta_gates_graph():
    g=_graph('DELTA_GATES',{'alpha_logits':('B','T','H'),'beta_logits':('B','T','H'),
              'dt_bias':('H',),'ssm_a':('H',)},['decay','beta'],'GatedDeltaNetBlock._attention')
    x=g.add(Node('shifted','ADD',['alpha_logits','dt_bias']))
    z=_const(g,'zero',0.0);one=_const(g,'one',1.0);minus=_const(g,'minus',-1.0)
    positive=g.add(Node('positive','MAXIMUM',[x,z]));neg=g.add(Node('negative','MUL',[x,minus]))
    absolute=g.add(Node('absolute','MAXIMUM',[x,neg]));negabs=g.add(Node('negabs','MUL',[absolute,minus]))
    ex=g.add(Node('ex','EXP',[negabs]));plus=g.add(Node('plus','ADD',[ex,one]));lg=g.add(Node('log','LOG',[plus]))
    sp=g.add(Node('softplus','ADD',[positive,lg]));la=g.add(Node('log_alpha','MUL',[sp,'ssm_a']))
    g.add(Node('decay','EXP',[la]));g.add(Node('beta','SIGMOID',['beta_logits']))
    g.metadata['ssm_a']='Already-transformed signed multiplier from source state, not an assumed A_log parameter.'
    g.validate();return g


def gated_delta_step_graph(key_dim=4, value_dim=3):
    _positive(key_dim=key_dim,value_dim=value_dim)
    g=_graph('GATED_DELTA_STEP',{'q':('B','H',1,key_dim),'k':('B','H',1,key_dim),
              'v':('B','H',value_dim,1),'beta':('B','H',1,1),'decay':('B','H',1,1),
              'state':('B','H',value_dim,key_dim)},['out','next_state'],'GatedDeltaNetBlock._attention')
    decayed=g.add(Node('decayed','MUL',['state','decay']))
    pred=g.add(Node('projected','MUL',[decayed,'k']))
    pred=g.add(Node('prediction','REDUCE_SUM',[pred],{'axis':-1,'keepdim':True}))
    error=g.add(Node('error','SUB',['v',pred]));delta=g.add(Node('delta','MUL',[error,'beta']))
    update=g.add(Node('update','MUL',[delta,'k']));g.add(Node('next_state','ADD',[decayed,update]))
    read=g.add(Node('read','MUL',['next_state','q']))
    g.add(Node('out','REDUCE_SUM',[read],{'axis':-1,'keepdim':False}))
    g.metadata['scope']='q is already L2-normalized and scaled by key_dim**-0.5; k normalized; decay already exponentiated.'
    g.validate();return g


def gated_delta_scan_graph(batch=1, heads=2, tokens=3, key_dim=4, value_dim=3):
    _positive(batch=batch,heads=heads,tokens=tokens,key_dim=key_dim,value_dim=value_dim)
    g=_graph('GATED_DELTA_SCAN',{'q':(batch,heads,tokens,key_dim),'k':(batch,heads,tokens,key_dim),
              'v':(batch,heads,tokens,value_dim),'beta':(batch,heads,tokens),
              'decay':(batch,heads,tokens,1),'state':(batch,heads,value_dim,key_dim)},['out','next_state'],
              'GatedDeltaNetBlock._attention')
    state='state';outs=[]
    for t in range(tokens):
        bindings={'state':state}
        for value in ('q','k','v','beta','decay'):
            s=_slice(g,f'{value}{t}',value,2,t,t+1)
            shape=(batch,heads,1,key_dim) if value in ('q','k') else (batch,heads,value_dim,1) if value=='v' else (batch,heads,1,1)
            bindings[value]=_reshape(g,f'{value}_step{t}',s,shape)
        out,state=inline(g,gated_delta_step_graph(key_dim,value_dim),f'step{t}',bindings)
        outs.append(_reshape(g,f'out{t}',out,(batch,heads,1,value_dim)))
    g.add(Node('out','CONCAT',outs,{'axis':2}));g.add(Node('next_state','IDENTITY',[state]))
    g.metadata['scope']='Fixed-length float32 scalar-decay reference scan. Reset means caller supplies zero state. No KDA per-channel decay or padding/runtime UOps.'
    g.validate();return g


def gated_rmsnorm_graph(eps=1e-6):
    g=_graph('GATED_RMSNORM',{'x':('...','D'),'weight':('D',),'gate':('...','D')},['out'],
              'GatedDeltaNetBlock._attention')
    norm=inline(g,rmsnorm_graph(eps),'norm',{'x':'x','weight':'weight'})[0]
    gate=inline(g,activation_graph('silu'),'gate',{'x':'gate'})[0]
    g.add(Node('out','MUL',[norm,gate]));g.validate();return g
