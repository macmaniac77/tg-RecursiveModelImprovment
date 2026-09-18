# Data-system destination

Data is another explicit research dimension.

The system should track:

- dataset identity/version/license;
- exact train/dev/test/hidden/OOD split;
- filtering and deduplication;
- augmentation;
- synthetic data generation;
- pseudo-label provenance;
- active-learning selection;
- label revisions;
- leakage checks;
- data quality metrics;
- task/head coverage.

Never allow the search process to silently contaminate held-out evaluation sets.

For self-improving loops, distinguish:
- real-world newly observed data;
- model-generated/simulated data;
- human-corrected data;
- replayed historical data.

A future “wake/dream” loop may alternate real data acquisition with offline replay/simulation, but provenance must remain explicit.
