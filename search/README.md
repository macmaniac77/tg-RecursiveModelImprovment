# Research/search loop destination

This is the meta-research layer.

It should decide which experiment to run next using the accumulated architecture graph, benchmark history, and available compute.

Expected stages:

- candidate generation;
- diversity/novelty pressure;
- cheap static filtering;
- proxy/short training;
- successive halving/promotion;
- full validation for finalists;
- replay of historical experiment trees;
- human proposal injection;
- preservation of divergent promising branches;
- periodic learned-policy/researcher updates.

The search policy must be separable from the experiment executor. Historical replay should permit testing alternative search policies without rerunning every expensive training job.

A future learned researcher may use RAG/graph memory continuously and periodically distill accumulated experiments into local weights/adapters. Exact experiment history remains authoritative.
