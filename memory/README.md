# Research memory destination

Use three distinct memory layers:

## 1. Exact symbolic memory
Architecture definitions, AOIs, graph diffs, specs, source mappings.

## 2. Empirical research memory
Experiment database: conditions, metrics, failures, lineage, observations.

## 3. Learned intuition
Optional periodically trained local model/LoRA/adapter that predicts promising mutations or summarizes patterns.

The first two are authoritative and lossless. The third is disposable/retrainable.

RAG/graph retrieval should provide working memory without bloating model context. Periodic training may consolidate repeated discoveries into a local researcher, but should never replace exact provenance.
