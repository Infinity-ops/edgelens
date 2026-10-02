"""
edgelens.benchmark.pipeline_loader
------------------------------------
Loads a user script for `edgelens benchmark --pipeline my_pipeline.py`.

The script defines ONE of these module-level functions (called once, before
timing starts — do one-time setup like opening a camera or loading a model
inside it):

    def build_pipeline():            # preferred (v0.1.0+)
        pipe = edgelens.Pipeline("bearing", pack="timeseries", ...)
        pipe.add_stage("filter", ...)
        return pipe

    def build_stage_fns():           # classic contract, still supported
        return {"capture": f1, "preprocess": f2, ...}

Since v0.1.0 a stage dict may use ANY stage names, in any number, in the
order they should run. The classic six vision names are recognised and get
the vision pack automatically; other names get the custom pack, with roles
inferred from the names (override with a Pipeline and role=...).
"""

import importlib.util
from pathlib import Path

ENTRYPOINTS = ("build_pipeline", "build_stage_fns")
ENTRYPOINT_NAME = "build_stage_fns"   # backward-compatible constant


def _load_module(script_path):
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
    return module


def _validate_stage_dict(stage_fns, where):
    if not isinstance(stage_fns, dict):
        raise RuntimeError(
            f"{where} must return a dict of stage functions (or an edgelens.Pipeline), "
            f"got {type(stage_fns).__name__}."
        )
    if not stage_fns:
        raise RuntimeError(f"{where} returned an empty dict — define at least one stage.")
    bad_keys = [k for k in stage_fns if not isinstance(k, str) or not k]
    if bad_keys:
        raise RuntimeError(f"{where}: stage names must be non-empty strings, got {bad_keys!r}.")
    non_callable = [k for k, v in stage_fns.items() if not callable(v)]
    if non_callable:
        raise RuntimeError(
            f"{where}: the following stage(s) are not callable: {', '.join(non_callable)}. "
            f"Each stage must be a function taking zero arguments, or one (the previous "
            f"stage's output)."
        )


def load_pipeline_from_script(script_path, pack=None):
    """Import the script and return an edgelens.Pipeline."""
    from ..core.pipeline import Pipeline

    module = _load_module(script_path)
    name = next((n for n in ENTRYPOINTS if getattr(module, n, None) is not None), None)
    if name is None:
        raise RuntimeError(
            f"'{script_path}' must define a top-level function `build_pipeline()` "
            f"(returning an edgelens.Pipeline) or `build_stage_fns()` (returning a "
            f"dict of stage functions). See tests/fixtures/example_pipeline.py."
        )
    entrypoint = getattr(module, name)
    if not callable(entrypoint):
        raise RuntimeError(
            f"'{name}' in '{script_path}' must be a function, not a "
            f"{type(entrypoint).__name__}."
        )
    try:
        built = entrypoint()
    except Exception as e:
        raise RuntimeError(
            f"'{name}()' in '{script_path}' raised an error while running:\n"
            f"{type(e).__name__}: {e}"
        ) from e

    where = f"'{name}()' in '{script_path}'"
    if isinstance(built, Pipeline):
        if len(built) == 0:
            raise RuntimeError(f"{where} returned a Pipeline with no stages.")
        return built
    _validate_stage_dict(built, where)
    try:
        return Pipeline.from_stage_fns(built, name=Path(script_path).stem, pack=pack)
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"{where}: {e}") from e


def load_stage_fns_from_script(script_path):
    """Backward-compatible: return {stage_name: callable} from the script."""
    pipe = load_pipeline_from_script(script_path)
    return {s.name: s.fn for s in pipe.stages}
