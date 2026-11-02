# Executable examples

Install the package before running scripts from the repository root.

## JEPA-style latent prediction

```bash
pip install -e '.[tinygrad]' 'tinygrad==0.14.0'
python examples/jepa.py --steps 20 --seed 7
```

The training example is tested with Tinygrad 0.14.0 and its current autodiff and
optimizer APIs; the command pins that runtime for reproducibility.

This small educational example uses aligned vector views and two-layer ReLU
MLPs. The context view hides the last four of eight input coordinates; the target
view contains the full vector. It predicts an **embedding**, not the missing input
values. There are no labels, dataset downloads or pretrained weights.

`aoi.jepa_graph()` expands three MLP instances to canonical primitives:

- `context_encoder` maps the partial view to a latent vector.
- `predictor` maps that latent to the predicted target embedding.
- `target_encoder` maps the full view to a stopped-gradient target embedding.
- The scalar loss is mean squared embedding error over batch and feature axes.

All weights and biases are graph inputs. The target starts as a separate copy of
the context encoder. Each step updates only context/predictor parameters with
SGD, then applies `aoi.ema_update_graph(decay=0.99)` to each target parameter using
the updated context parameters. The EMA graph returns detached values; the
example performs the assignments explicitly. `STOP_GRADIENT` is a real primitive
with Tinygrad `detach()` semantics, preserved across graph serialization.

The script prints seed, configuration, Tinygrad version, device and loss values.
The last reported loss is measured **before** the last optimizer/EMA update.
It uses a fixed synthetic batch for an execution smoke test. A decreasing loss
can reflect representational collapse and does not establish useful features,
generalization, or performance on a downstream task.

This is an original, simplified JEPA-style example inspired by the
[I-JEPA training design](https://github.com/facebookresearch/ijepa/blob/main/src/train.py).
It is **not an equivalent implementation of I-JEPA or V-JEPA**: it has no ViT,
patch masking/sampling, spatial positions, multiple target regions, target
LayerNorm, Smooth L1 loss, video path, checkpoint loader or large-scale recipe.
Those require separate architecture definitions and equivalence evidence.

Export and inspect the primitive graph with the existing viewer:

```bash
python examples/jepa.py --steps 3 --graph artifacts/jepa.arch.json
pip install -e '.[viewer]'
python -m tgaoi.viewer --graph artifacts/jepa.arch.json --output artifacts/jepa-viewer.html
```

Numerical and semantic evidence lives in `tests/test_jepa.py`: NumPy forward
equivalence, finite-difference online gradients, blocked target gradients,
EMA values/endpoints/detachment, JSON round-trip and a short training smoke test.

## Other examples

- `build_attention.py`: construct the canonical attention graph.
- `torch_fx_to_tinygrad.py`: translate a small PyTorch module to IR and execute
  it through Tinygrad (requires the `torch` extra).
