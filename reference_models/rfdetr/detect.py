"""PLACEHOLDER ONLY.

Replace or wrap this file with the user's real Tinygrad RF-DETR detect.py.

Required future contract:

    build_model(package_config) -> model
    load_weights(model, checkpoint_path) -> model
    preprocess(raw_input) -> tensor(s)
    predict(model, inputs) -> raw_outputs
    postprocess(raw_outputs) -> detections

The generic harness should eventually be able to call a stable adapter without
changing the supplied model's inference semantics.

Do not implement fake RF-DETR logic here.
"""


def _placeholder():
    raise NotImplementedError(
        "Waiting for the real Tinygrad RF-DETR detect.py from the user. "
        "Read AGENT_HANDOFF.md before replacing this placeholder."
    )


if __name__ == "__main__":
    _placeholder()
