# Experiment definitions

This directory should contain declarative experiment/search configurations.

A cascade experiment should specify:

- canonical parent model;
- task and dataset;
- train/validation/hidden/OOD partitions;
- compute budget;
- allowed mutation families;
- whether architecture, training, or data may change;
- promotion schedule;
- objective gates;
- metric tolerances;
- reproducibility seeds;
- artifact/checkpoint policy.

Example conceptual structure:

```yaml
parent: artifacts/models/rfdetr_baseline

budget:
  device: RTX4070
  max_wall_hours: 12

search:
  mutations:
    - attention
    - recurrent_depth
    - latent_feedback
    - normalization
  proxy_steps: 1000
  promotion_fraction: 0.2

cascade:
  - require: {coco_ap: ">= baseline"}
  - minimize: latency_ms
    preserve: [coco_ap]
  - minimize: time_to_target_ap
    preserve: [coco_ap, latency_ms]
  - minimize: peak_vram_mb
    preserve: [coco_ap, latency_ms, time_to_target_ap]
```

The search policy interprets this file; it should not hard-code one task or benchmark.
