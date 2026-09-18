# Checkpoint ingest rules

## Supported checkpoint classes to detect

For PyTorch inputs, inspect before loading into the architecture pipeline:

- raw `state_dict`;
- dictionary with `model` or `state_dict`;
- full serialized `nn.Module`;
- TorchScript;
- torch.export ExportedProgram;
- project-specific checkpoint.

Security note: ordinary `torch.load` may deserialize Python objects. Treat unknown checkpoints as untrusted input and prefer restricted/safe loading modes where available.

## Architecture recovery priority

1. exported graph/program embedded in package;
2. explicit manifest/config + known model code;
3. registered architecture family + exact config;
4. parameter-signature match only as an assistive hint;
5. otherwise stop.

Parameter shapes alone are not an authoritative architecture description.

## Parameter mapping

Mapping should use, in descending confidence:

1. exact canonical provenance established during graph import;
2. exact framework name/path mapping;
3. structural graph correspondence + shape;
4. human-reviewed mapping rule.

Ambiguous mappings must not be silently accepted.

## Equivalence

Before declaring a model imported:

- run same source inputs through source framework and Tinygrad canonical form;
- compare all meaningful outputs;
- compare intermediate activations when debugging mismatch;
- record numerical tolerance and dtype;
- test more than one input where the model contains data-dependent behavior.

Only after equivalence is established should benchmarking/search begin.
