"""
edgelens.benchmark.pipeline_loader
------------------------------------
Loads a user-supplied Python script and extracts the six pipeline stage
functions from it, for `edgelens benchmark --pipeline my_pipeline.py`.

This is the general-purpose alternative to `--model`: where `--model`
only works for a plain ONNX forward pass, `--pipeline` lets you wire in
a real camera, real preprocessing, a non-ONNX runtime, or anything else
`stage_fns` already supported from Python — but from the CLI, as a
file path, with no edgelens import required in your script.

CONTRACT the script must satisfy — define a module-level function:

    def build_stage_fns():
        # Any one-time setup goes here: open a camera, load your real
        # model, allocate buffers, etc. Called exactly once, before
        # timing starts.
        ...
        return {
            "capture": capture_fn,
            "preprocess": preprocess_fn,
            "h2d_copy": h2d_copy_fn,
            "inference": inference_fn,
            "d2h_copy": d2h_copy_fn,
            "postprocess": postprocess_fn,
        }

Each value must be a zero-argument callable. EdgeLens times each one
individually, in order, once per benchmark iteration. See
tests/fixtures/example_pipeline.py for a minimal working example.
"""

import importlib.util
from pathlib import Path

REQUIRED_STAGES = ("capture", "preprocess", "h2d_copy", "inference", "d2h_copy", "postprocess")
ENTRYPOINT_NAME = "build_stage_fns"


def load_stage_fns_from_script(script_path):
    """Dynamically imports `script_path` and calls its build_stage_fns()
    to get the six stage functions.

    This is a user-facing contract, not an internal API — every failure
    mode below raises a specific, actionable RuntimeError rather than
    letting a raw traceback (ImportError, AttributeError, TypeError...)
    surface, since the person reading the error is debugging their own
    script, not EdgeLens' internals.
    """
    path = Path(script_path)
    if not path.exists():
        raise RuntimeError(f"Pipeline script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("edgelens_user_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load '{script_path}' as a Python module.")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(
            f"Error while importing '{script_path}':\n{type(e).__name__}: {e}"
        ) from e

    entrypoint = getattr(module, ENTRYPOINT_NAME, None)
    if entrypoint is None:
        raise RuntimeError(
            f"'{script_path}' must define a top-level function "
            f"`{ENTRYPOINT_NAME}()` that returns a dict of the six stage "
            f"functions. See tests/fixtures/example_pipeline.py for an "
            f"example."
        )
    if not callable(entrypoint):
        raise RuntimeError(
            f"'{ENTRYPOINT_NAME}' in '{script_path}' must be a function, "
            f"not a {type(entrypoint).__name__}."
        )

    try:
        stage_fns = entrypoint()
    except Exception as e:
        raise RuntimeError(
            f"'{ENTRYPOINT_NAME}()' in '{script_path}' raised an error "
            f"while running:\n{type(e).__name__}: {e}"
        ) from e

    if not isinstance(stage_fns, dict):
        raise RuntimeError(
            f"'{ENTRYPOINT_NAME}()' in '{script_path}' must return a dict, "
            f"got {type(stage_fns).__name__}."
        )

    missing = [s for s in REQUIRED_STAGES if s not in stage_fns]
    if missing:
        raise RuntimeError(
            f"'{ENTRYPOINT_NAME}()' in '{script_path}' is missing required "
            f"stage(s): {', '.join(missing)}.\n"
            f"Required stages: {', '.join(REQUIRED_STAGES)}."
        )

    extra = [k for k in stage_fns if k not in REQUIRED_STAGES]
    if extra:
        raise RuntimeError(
            f"'{ENTRYPOINT_NAME}()' in '{script_path}' returned unexpected "
            f"key(s): {', '.join(extra)}.\n"
            f"Required stages: {', '.join(REQUIRED_STAGES)}."
        )

    non_callable = [k for k in REQUIRED_STAGES if not callable(stage_fns[k])]
    if non_callable:
        raise RuntimeError(
            f"'{ENTRYPOINT_NAME}()' in '{script_path}': the following "
            f"stage(s) are not callable: {', '.join(non_callable)}. Each "
            f"stage must be a zero-argument function."
        )

    return stage_fns