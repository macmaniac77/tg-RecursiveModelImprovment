# Training-system destination

Training is a first-class search dimension, separate from model topology.

This subsystem should describe and eventually allow controlled variation of:

- pretraining objectives;
- supervised/post-training objectives;
- RL objectives;
- distillation;
- self-training / pseudo-labeling;
- active learning;
- self-play / simulated replay;
- curriculum and data ordering;
- augmentation;
- optimizer choice;
- learning-rate schedules;
- regularization;
- multi-task weighting;
- auxiliary losses;
- checkpointing/consolidation.

Training recipes should themselves be serializable/versioned objects with provenance, so the research loop can distinguish “architecture improvement” from “training improvement.”

Long term, architecture and training may co-evolve, but their mutations and attribution should remain separable.
