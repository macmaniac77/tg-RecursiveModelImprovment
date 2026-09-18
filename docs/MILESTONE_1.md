# Milestone 1 — PyTorch MLP to Tinygrad equivalence

Target:

```text
PyTorch MLP
   ↓ torch.fx
canonical typed IR
   ↓
Tinygrad backend
   ↓
same weights + same input
   ↓
numerically equivalent output
```

The reference model is `Linear(4,7) -> ReLU -> Linear(7,3)`.

## Translation

`torch.nn.Linear` is expanded into:

```text
weight
  ↓ TRANSPOSE
input + transposed weight
  ↓ MATMUL
bias
  ↓ ADD
```

`torch.nn.ReLU` is expanded into:

```text
x + CONST(0)
  ↓ MAXIMUM
```

Model parameters are explicit graph inputs using names such as
`param::fc1.weight`. They are not retained as hidden PyTorch objects.

Unsupported FX nodes remain `OPAQUE::<target>`; the equivalence test requires
there to be no opaque nodes in the reference MLP.

## Success criterion

`tests/test_torch_fx_tinygrad_equivalence.py` compares both paths using the
same initialized weights and input with:

```python
np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
```

GitHub Actions installs CPU PyTorch + Tinygrad and runs the complete suite.
