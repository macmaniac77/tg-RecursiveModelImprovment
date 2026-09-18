# Reference model packages

Reference models are real model implementations used to prove the generic architecture/search contract.

Each reference package should contain or point to:

- inference code;
- training code;
- checkpoint(s);
- package manifest;
- model dictionary;
- instance architecture map;
- benchmark adapter;
- baseline metrics;
- canonical architecture artifact after ingest.

A reference model is not allowed to leak model-specific assumptions into the core search engine.

The first reference organism is `rfdetr/`.
