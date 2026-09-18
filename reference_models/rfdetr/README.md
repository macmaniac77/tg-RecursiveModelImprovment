# RF-DETR reference package

This directory is reserved for the first full-scale reference model used to prove the generic research loop.

The user intends to provide a Tinygrad-converted RF-DETR implementation.

## Expected role

RF-DETR should prove that the repository can take a real model implementation and turn it into:

- a canonical AOI hierarchy;
- a parameter/weight map;
- a generic inference contract;
- a generic training contract;
- a generic benchmark contract;
- a mutable architecture package for automated research.

RF-DETR must remain a **consumer of the generic architecture/search system**, not become hard-coded into core tg-aoi logic.

## Placeholder files

The placeholder files in this directory describe the expected interfaces. Replace them with or adapt them around the real implementation when supplied.

See `AGENT_HANDOFF.md` first.
