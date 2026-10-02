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


def test_any_stage_names_are_allowed_since_v010(tmp_path):
    # v0.1.0: the six vision names are no longer mandatory. A sensor
    # pipeline with its own stage names (and count) loads and runs.
    script = _write(tmp_path, "sensor.py", (
        "def build_stage_fns():\n"
        "    f = lambda: None\n"
        "    return {'acquire': f, 'filter': f, 'fft': f, 'inference': f, 'decision': f}\n"
    ))
    fns = load_stage_fns_from_script(script)
    assert list(fns) == ["acquire", "filter", "fft", "inference", "decision"]
    result = run_benchmark(iterations=5, warmup=1, pipeline_path=script)
    assert [s["name"] for s in result["pipeline"]["stages"]] == list(fns)
    assert result["pipeline"]["pack"] == "custom"
    roles = {s["name"]: s["role"] for s in result["pipeline"]["stages"]}
    assert roles == {"acquire": "input", "filter": "preprocess", "fft": "preprocess",
                     "inference": "inference", "decision": "decision"}


def test_classic_six_stage_dict_selects_vision_pack():
    result = run_benchmark(iterations=5, warmup=1, pipeline_path=EXAMPLE_PIPELINE)
    assert result["pipeline"]["pack"] == "vision"


def test_empty_stage_dict_is_rejected(tmp_path):
    script = _write(tmp_path, "empty.py", "def build_stage_fns():\n    return {}\n")
    with pytest.raises(RuntimeError, match="empty dict"):
        load_stage_fns_from_script(script)


def test_build_pipeline_entrypoint_returns_pipeline(tmp_path):
    script = _write(tmp_path, "pipe.py", (
        "import edgelens as el\n"
        "def build_pipeline():\n"
        "    p = el.Pipeline('ts', pack='timeseries', sample_rate_hz=1000, window=100, hop=50)\n"
        "    p.add_stage('filter', lambda: None)\n"
        "    p.add_stage('inference', lambda: None)\n"
        "    return p\n"
    ))
    result = run_benchmark(iterations=5, warmup=1, pipeline_path=script)
    assert result["pipeline"]["pack"] == "timeseries"
    assert result["requirements"]["deadline_ms"] == 50.0   # hop period by default


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