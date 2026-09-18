# Benchmark system destination

Benchmarks define measurable missions for the research loop.

Support multiple objective styles:

- classification accuracy/loss;
- language validation loss / time-to-target-loss;
- COCO AP and detection metrics;
- segmentation metrics;
- depth error;
- latency;
- peak VRAM;
- training FLOPs/time;
- parameter count;
- energy where measurable;
- robustness/OOD metrics.

Prefer **successive constraints / Pareto fronts** over collapsing everything into one arbitrary scalar.

Example:

1. meet or beat baseline AP;
2. while preserving AP, reduce inference latency;
3. while preserving both, reduce training compute;
4. while preserving all prior constraints, reduce VRAM.

Maintain hidden/OOD validation that the search policy cannot directly optimize against where feasible.
