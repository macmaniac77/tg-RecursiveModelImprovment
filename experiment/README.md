# Experiment harness destination

This subsystem should run reproducible architecture/training experiments on constrained hardware.

Every run should record at minimum:

- architecture graph hash and full graph;
- code/commit hash;
- dataset/version/split;
- training objective(s);
- optimizer and schedule;
- seed(s);
- hardware/device/software versions;
- parameter count;
- training tokens/examples/steps;
- wall-clock time;
- FLOPs or best available compute estimate;
- peak VRAM;
- train/validation curves;
- final task metrics;
- failure/instability reason;
- parent experiment and mutation provenance.

The first target is RTX-4070-class research: many small, controlled experiments, early stopping, and promotion of promising candidates rather than frontier-scale training.

Results should be machine-queryable so an agent can answer questions such as “what happened the last 20 times latent feedback was added under a 50M-parameter budget?”
