import os

import pytest

from edgelens.benchmark.pipeline_loader import load_stage_fns_from_script
from edgelens.benchmark.runner import run_benchmark

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
EXAMPLE_PIPELINE = os.path.join(FIXTURES, "example_pipeline.py")


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return str(path)


def test_example_pipeline_loads_and_runs():
    fns = load_stage_fns_from_script(EXAMPLE_PIPELINE)
    assert set(fns.keys()) == {
        "capture", "preprocess", "h2d_copy", "inference", "d2h_copy", "postprocess",
    }
    for fn in fns.values():
        fn()  # each stage should run without error


def test_benchmark_end_to_end_with_pipeline_script():
    result = run_benchmark(iterations=10, warmup=2, pipeline_path=EXAMPLE_PIPELINE)
    assert result["mode"] == "hardware"
    assert "custom pipeline" in result["pipeline_source"]
    assert EXAMPLE_PIPELINE in result["pipeline_source"]
    assert result["total_latency_ms"] > 0


def test_missing_file_raises_clear_error():
    with pytest.raises(RuntimeError, match="not found"):
        load_stage_fns_from_script("/no/such/file.py")


def test_missing_entrypoint_function(tmp_path):
    script = _write(tmp_path, "bad1.py", "x = 1\n")
    with pytest.raises(RuntimeError, match="build_stage_fns"):
        load_stage_fns_from_script(script)


def test_entrypoint_not_callable(tmp_path):
    script = _write(tmp_path, "bad2.py", "build_stage_fns = 42\n")
    with pytest.raises(RuntimeError, match="must be a function"):
        load_stage_fns_from_script(script)


def test_entrypoint_raises_is_wrapped(tmp_path):
    script = _write(tmp_path, "bad3.py",
                     "def build_stage_fns():\n    raise ValueError('boom')\n")
    with pytest.raises(RuntimeError, match="raised an error"):
        load_stage_fns_from_script(script)


def test_entrypoint_returns_non_dict(tmp_path):
    script = _write(tmp_path, "bad4.py",
                     "def build_stage_fns():\n    return [1, 2, 3]\n")
    with pytest.raises(RuntimeError, match="must return a dict"):
        load_stage_fns_from_script(script)


def test_entrypoint_missing_required_stage(tmp_path):
    script = _write(tmp_path, "bad5.py", (
        "def build_stage_fns():\n"
        "    f = lambda: None\n"
        "    return {'capture': f, 'preprocess': f, 'h2d_copy': f, "
        "'inference': f, 'd2h_copy': f}\n"  # missing postprocess
    ))
    with pytest.raises(RuntimeError, match="missing required stage"):
        load_stage_fns_from_script(script)


def test_entrypoint_extra_key(tmp_path):
    script = _write(tmp_path, "bad6.py", (
        "def build_stage_fns():\n"
        "    f = lambda: None\n"
        "    return {'capture': f, 'preprocess': f, 'h2d_copy': f, "
        "'inference': f, 'd2h_copy': f, 'postprocess': f, 'extra_stage': f}\n"
    ))
    with pytest.raises(RuntimeError, match="unexpected key"):
        load_stage_fns_from_script(script)


def test_entrypoint_non_callable_stage_value(tmp_path):
    script = _write(tmp_path, "bad7.py", (
        "def build_stage_fns():\n"
        "    f = lambda: None\n"
        "    return {'capture': f, 'preprocess': f, 'h2d_copy': f, "
        "'inference': 123, 'd2h_copy': f, 'postprocess': f}\n"  # inference not callable
    ))
    with pytest.raises(RuntimeError, match="not callable"):
        load_stage_fns_from_script(script)


def test_import_error_in_script_is_wrapped(tmp_path):
    script = _write(tmp_path, "bad8.py", "import this_module_does_not_exist\n")
    with pytest.raises(RuntimeError, match="Error while importing"):
        load_stage_fns_from_script(script)


def test_model_and_pipeline_together_is_ambiguous():
    with pytest.raises(RuntimeError, match="Ambiguous pipeline source"):
        run_benchmark(iterations=5, model_path="whatever.onnx",
                      pipeline_path=EXAMPLE_PIPELINE)