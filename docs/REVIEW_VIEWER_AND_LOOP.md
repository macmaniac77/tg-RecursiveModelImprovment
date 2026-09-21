# Primitive, viewer, and measured-loop review

Reviewed baseline: `39fe69a0579dbd922d12e0778f842a9fea66ac37` (2026-09-21).

## Findings, in priority order

1. **No measured architecture-improvement loop exists yet.**
   `reference_models/rfdetr/benchmark_adapter.py::evaluate_model` raises
   `NotImplementedError`. `docs/RESEARCH_LOOP.md`, `research/benchmark_spec.json`,
   and `experiments/example_cascade.yaml` describe the desired cascade, but no
   executable candidate runner, baseline comparator, promotion gate, or experiment
   ledger implements it. Do not interpret the training script as this loop.
2. **RF-DETR training has a detached GIoU loss.** In
   `reference_models/rfdetr/train.py::SetCriterion._loss_boxes`,
   `Tensor(np.diag(giou.numpy()))` severs the gradient path from GIoU to predicted
   boxes. Its scalar can be logged while contributing no box gradient. Replace
   this with differentiable tensor diagonal selection and verify against an
   independent gradient reference before trusting training comparisons. This
   review does not modify or claim to validate the full RF-DETR trainer.
3. **Most AOIs are structurally valid sketches, not executable translations.**
   There are 28 exported builders, 15 backend primitive handlers, and five default
   AOIs with complete direct primitive coverage. ATTENTION cannot execute its
   nested SOFTMAX. Many higher blocks omit explicit weight inputs: e.g.
   MS_DEFORM_ATTN calls LINEAR with only one input; DETR_SET_CRITERION supplies
   three inputs to the default four-input matcher. ViT attention lacks explicit
   multihead reshape/merge and is not source-equivalence validated. The viewer
   exposes these gaps rather than implying recursive closure.
4. **IR validation was incomplete even for identity integrity.** Graphs loaded
   through `from_dict` could contain duplicate IDs that overwrite values at
   execution. Fixed and regression-tested. The validator still does not enforce
   general op arity, shape propagation, symbolic constraints, dtype compatibility,
   or AOI binding validity; those remain promotion prerequisites.
5. **ADAM_STEP is not standard Adam.** It updates moments and uses uncorrected
   moments in the parameter update. It has no timestep/bias correction or AdamW
   decay. Added explicit metadata/docstring; preserved existing mathematics.
6. **The RF-DETR package manifest remains stale.** `package.yaml` points at
   `weights.pt` with unknown format, while the repo includes an LFS-tracked
   `weights/rfdetr_nano_tinygrad.safetensors`. Exact config, input dimensions,
   weight compatibility and hashes need a runtime baseline check before updating
   this into an authoritative experiment manifest.
7. **Training loss/time output is insufficient experiment evidence.**
   `train_steps` prints loss and mean wall seconds, then the CLI saves weights;
   it does not produce held-out COCO AP, synchronized inference distributions,
   peak memory measurements, exact mutation/state inheritance, or gate decisions.
   Partial gradient accumulation is also dropped when steps is not divisible by
   the accumulation interval. A loss decrease alone cannot promote a candidate.

## What this change implements

A local read-only HTML viewer, generated directly from the RF-DETR mapping,
dictionary and canonical AOI graph definitions. It supports source inspection,
expandable mapped and semantic hierarchies, parameter declarations, searchable
capability diagnostics, optional canonical JSON graphs, and a loop-status review.
It does not import the RF-DETR model or claim runtime parameter counts. The
block overview is a labeled composition projection, not invented dataflow.

## Smallest trustworthy measured loop to implement next

1. Freeze one parent manifest: code revision, canonical architecture hash,
   checkpoint hash, package config, framework/compiler version, dataset manifest,
   fixed search-visible validation IDs, seed, hardware and precision.
2. Make the benchmark adapter execute the unchanged parent first: COCO AP,
   warmed/synchronized repeated inference timings, memory, and parameter count.
   Persist raw measurements, protocol, failures, and budget, not just summaries.
3. Record one mutation as an immutable child: parent ID, exact graph/code/config
   diff, rationale, weight inheritance mapping, optimizer initialization and budget.
4. Fail closed on unresolved operations/bindings, invalid shapes, nonfinite
   outputs/gradients, missing metrics, crashes, or incomparable protocols.
5. Evaluate parent and child on the same protocol/seeds. Require a declared quality
   tolerance before a latency improvement can count; then preserve quality and
   latency when optimizing training time/memory. Gate missing/NaN data as failure.
   Use repeat variance and predeclared thresholds to avoid promoting timing noise.
6. Append a result record containing raw metrics, deltas, per-gate decisions,
   rejection reasons and artifact hashes. A human reviews this exact evidence
   before any production/champion replacement. Hidden/OOD validation remains
   separate from search-visible scores.

Only after one reproducible parent/child comparison should the system queue many
mutations or call itself recursive improvement. The viewer can later consume the
ledger; it should not invent decisions independently.

## Validation

- `DEV=PYTHON python -m pytest -q`: numerical core and structural/viewer tests.
- Tinygrad 0.14.0 Python interpreter backend used because this environment exposes
  no usable native device by default. These checks are not GPU benchmarks.
- PyTorch importer equivalence test skips when PyTorch is absent; no RF-DETR
  numerical equivalence, training, COCO evaluation, or throughput claim is made.

- Final suite result: **15 passed, 1 skipped**. Generated JavaScript passes
  `node --check`; browser interaction/visual checks remain unverified because
  Chromium was unavailable and its download timed out.
