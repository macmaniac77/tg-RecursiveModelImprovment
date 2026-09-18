"""RF-DETR benchmark adapter placeholder.

When the real model is supplied, this module should become the narrow adapter
between the generic benchmark harness and RF-DETR-specific evaluation.

Expected future API:

    prepare_validation(package_config)
    evaluate_model(model, validation_data) -> dict[str, float]
    measure_inference(model, sample_batch) -> dict[str, float]
    smoke_test_training(train_adapter) -> dict[str, float]

At minimum return task quality + latency + memory metrics with enough metadata
to reproduce the measurement.

Do not hard-code benchmark logic into the architecture search engine.
"""


def evaluate_model(*args, **kwargs):
    raise NotImplementedError("Populate after the real RF-DETR implementation is supplied.")
