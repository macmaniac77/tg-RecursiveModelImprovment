# Qwen-capable Tinygrad LLM blocks

## Source and identity

Tinygrad **does provide `qwen3.8:27b`** in `tinygrad/llm/cli.py`, pointing at
`Qwen3.8-27B-IQ4_XS.gguf`. The pinned source is
`b7ff6dedf7e385b603bf9442ee23c6adf6601fb4`.
Its shared implementation is `tinygrad/llm/model.py`, not a standalone example.
The earlier catalog scoped only `examples/` and `extra/models/`; this change
extends it to `tinygrad/llm/` (including the AMD kernels) without changing the pin.

The exact 27B GGUF metadata, layer mix, weights and quality have **not** been
loaded or verified here. Block extraction follows the shared Qwen-capable source
paths, not an invented configuration for that checkpoint. In particular, the
loader chooses hybrid/SSM branches from GGUF architecture/config metadata.

## New reusable definitions

All are in `src/tgaoi/aoi/llm.py`, exported through `tgaoi.aoi` and visible in the
AOI catalog. They lower through ordinary Tinygrad tensor primitives.

| AOI | Source and semantics | Main decomposition |
|---|---|---|
| ROTARY_FREQUENCIES | `precompute_freqs_cis`; explicit positions, fixed dimension/theta | inverse-frequency constants, multiply, cos, sin |
| ROPE_HALF_SPLIT | `apply_rope`; even partial or full rotary prefix | slices, multiplies, add/subtract, concatenate untouched suffix |
| L2_NORMALIZE | DeltaNet Q/K normalization | square, sum, sqrt, maximum(epsilon), divide |
| MASKED_GQA | `TransformerBlock._attention` math | repeat each KV head, transpose, matmul, scale, explicit additive mask, stable softmax, matmul |
| KV_APPEND | Pure valid-prefix cache append | concatenate old/new K and V; return both updated arrays |
| CAUSAL_DEPTHWISE_CONV_STATE | DeltaNet input convolution | old state + projected inputs, per-tap multiply/add, SiLU, next-state slice |
| DELTA_GATES | DeltaNet scalar per-head gates | stable softplus(alpha + dt), multiply signed ssm_a, exp; sigmoid(beta) |
| GATED_DELTA_STEP | DeltaNet fallback recurrence | decay state, predict value, beta-scaled error, rank-one update, query readout |
| GATED_DELTA_SCAN | Fixed-token recurrence composition | namespaced repeated STEP with explicit carried state |
| GATED_RMSNORM | DeltaNet output gate | RMSNorm × SiLU(gate) |

Existing RMSNorm, bias-free SwiGLU, linear, residual and activation definitions
are reused. The source FFN mapping links **only its dense branch** to SWIGLU;
MoE routing/expert selection is still unresolved. Method mappings are collections
of reusable subgraphs, not claims that the entire source method is translated.

## State, layout and numerical contracts

- RoPE uses **half-split** pairing. Cos/sin are `[T, rotary_dim/2]`, inputs
  `[B,H,T,D]`; partial rotation preserves the tail. This is not interleaved RoPE.
- GQA repeats heads as `[k0,k0,...,k1,k1,...]`. Its mask is supplied explicitly.
  For a valid cached prefix of length P, query i can attend to key j iff
  `j <= P+i`. Entirely masked rows are outside the contract. No dropout.
- KV_APPEND consumes only the valid old prefix. Reset, bounded capacity,
  start-position validation and half-precision cache casts belong to the caller.
  It does not replicate an in-place full-capacity cache store.
- Convolution takes projected QKV values `[B,T,C]`, history `[B,K-1,C]`, weights
  `[C,K]`, and returns activated outputs plus raw next history. Kernel >= 2.
- DELTA_GATES consumes the signed multiplier named `ssm_a` in the source;
  it does not assume an A_log convention or introduce another negation/exp.
- STEP takes pre-normalized K and normalized/scaled Q. State is `[B,H,V,K]`;
  Q/K `[B,H,1,K]`, V `[B,H,V,1]`, beta/decay `[B,H,1,1]`.
- SCAN accepts Q/K `[B,H,T,K]`, V `[B,H,T,V]`, beta `[B,H,T]`, decay
  `[B,H,T,1]`, and explicit initial state. It returns `[B,H,T,V]` plus final
  state (source uses a transpose to `[B,T,H,V]` before output normalization).
  It unrolls a fixed token count. Passing zero state implements reset.
- KDA per-channel decay, symbolic token padding, fused AMD prefill kernels,
  mixed/half precision behavior and end-to-end quantized execution are excluded.

## Actual Tinygrad UOps

`TinygradBackend.inspect_uops(graph, feeds)` returns a JSON-serializable DAG with
operation names, dtypes, dependencies and output IDs. Its stage is explicitly
`lazy_tensor_uops_before_scheduling`. Buffer values/addresses are omitted.
This permits inspection of the real UOps generated from an AOI, but **does not**
claim to be scheduled/fused kernel IR, machine instructions or a speed benchmark.
The current viewer's graph inspector consumes canonical Graph JSON; this new
UOp-DAG schema is programmatic and is not yet a separate viewer panel.

The source's `UOp.store` and `UOp.after` encode mutation/order. Reference AOIs use
explicit returned state instead, which is easier to compare and reconstruct.
A backend-specific fused implementation can later replace the reference path
only after equivalence and performance validation.

## Viewer and validation

```bash
python -m tgaoi.viewer --examples reference_models/tinygrad/catalog.json.gz --output artifacts/tinygrad-viewer.html
```

Select **Tinygrad examples**, search `tinygrad/llm/model.py`, then inspect
`apply_rope`, `TransformerBlock._attention`, or `GatedDeltaNetBlock._attention`.
Their reusable-block buttons open the corresponding library graph.

Validation: **49 passed, 1 skipped** with `DEV=PYTHON python -m pytest -q` on
Tinygrad 0.14.0. Tests cover source-excerpt rotary/recurrence equivalence,
Tinygrad GQA equivalence for prefill and cached chunks, NumPy gate/convolution
references, serialization round trips, state continuity, no-op recurrence,
configuration rejection and actual UOp-DAG inspection. The skipped test needs
PyTorch. No full Qwen checkpoint, GPU benchmark or full-model quality claim.

DOM interaction checks also passed for navigating from the DeltaNet source method
to GATED_DELTA_SCAN in the viewer. Real-browser layout remains unverified.
