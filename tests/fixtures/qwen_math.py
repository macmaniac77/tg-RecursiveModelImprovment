# Tinygrad b7ff6dedf7e385b603bf9442ee23c6adf6601fb4; see TINYGRAD_LICENSE.
# Exact rotary function excerpts; recurrence loop lifted into a pure-state wrapper.
from tinygrad import Tensor

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def reference_delta(q,k,v,beta,alpha,state):
  T_pad = q.shape[2]
  q, k, v, beta = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
  alpha = alpha.unsqueeze(-1)
  outs = []
  for t in range(T_pad):
    s1 = state * alpha[:, :, t]  # decay the state
    delta = (v[:, :, t] - (s1*k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]  # the delta rule update
    state = s1 + delta * k[:, :, t]
    outs.append((state * q[:, :, t]).sum(-1))

  return outs[0].stack(*outs[1:],dim=1).transpose(1,2), state
