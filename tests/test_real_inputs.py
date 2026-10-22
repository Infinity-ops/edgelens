"""--model with real inputs, labels and a thread count.

From SML's motor-dropout LSTM on an AGX Orin: windows prepared by a partner
(raw float32, [N, 100, 150] for a [1, 100, 150] input) plus a regression
target needed hand-written replay scripts; synthetic inputs cannot say how
accurate a model is; and onnxruntime's thread count changed latency, CPU and
energy enough to belong in the result. Uses the repository's LSTM fixture
([1, 50, 16] -> [1, 1]) so no extra packages are needed.
"""
import json
from pathlib import Path

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")

from edgelens.benchmark import real_inputs as ri  # noqa: E402
from edgelens.benchmark.runner import run_benchmark  # noqa: E402
from edgelens.compare.engine import requirements_diff  # noqa: E402
from edgelens.validate.engine import validate  # noqa: E402

MODEL = str(Path(__file__).parent / "fixtures" / "lstm_anomaly.onnx")
SHAPE = (1, 50, 16)
N = 37


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((N, 50, 16)).astype(np.float32)        # no batch dim, like SML's
    y = np.repeat(np.array([1.0, 0.6, 0.3], np.float32), [15, 12, 10])
    paths = {"npy": tmp_path / "x.npy", "bin": tmp_path / "x.bin", "npz": tmp_path / "x.npz",
             "y": tmp_path / "y.bin", "ynpy": tmp_path / "y.npy"}
    np.save(paths["npy"], x)
    x.tofile(paths["bin"])
    np.savez(paths["npz"], window=x[:, None])
    y.tofile(paths["y"])
    np.save(paths["ynpy"], y)
    return x, y, {k: str(v) for k, v in paths.items()}


def _reference_predictions(x):
    s = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    return np.array([float(s.run(None, {name: x[k:k + 1]})[0].ravel()[0]) for k in range(len(x))])


# ---- loading ----

@pytest.mark.parametrize("fmt", ["npy", "bin", "npz"])
def test_every_format_gives_the_same_samples(data, fmt):
    x, _, p = data
    r = ri.RealInputs(p[fmt], [("window", SHAPE, np.float32)])
    assert r.n == N
    s = r.sample(5)["window"]
    assert s.shape == SHAPE and s.flags["C_CONTIGUOUS"] and np.array_equal(s[0], x[5])


def test_raw_file_of_the_wrong_size_says_why(data, tmp_path):
    bad = tmp_path / "bad.bin"
    np.zeros(50 * 16 + 3, np.float32).tofile(bad)
    with pytest.raises(ri.InputError, match="not a whole number"):
        ri.RealInputs(str(bad), [("window", SHAPE, np.float32)])


def test_wrong_shape_names_the_expected_one(data, tmp_path):
    f = tmp_path / "w.npy"
    np.save(f, np.zeros((4, 50, 15), np.float32))
    with pytest.raises(ri.InputError, match=r"\[N, 1, 50, 16\] or \[N, 50, 16\]"):
        ri.RealInputs(str(f), [("window", SHAPE, np.float32)])


def test_labels_must_match_the_samples(data, tmp_path):
    f = tmp_path / "short.npy"
    np.save(f, np.zeros(N - 1))
    with pytest.raises(ri.InputError, match="one label per sample"):
        ri.load_labels(str(f), N)


def test_spread_order_visits_every_sample_once_per_pass():
    o = ri.SampleOrder(2437, "spread")
    assert len({o.index(i) for i in range(2437)}) == 2437 and o.stride > 1
    assert [ri.SampleOrder(5).index(i) for i in range(6)] == [0, 1, 2, 3, 4, 0]


# ---- running ----

def test_accuracy_matches_an_independent_loop(data):
    x, y, p = data
    r = run_benchmark(iterations=N, warmup=0, model_path=MODEL, inputs=p["bin"],
                      labels=p["y"], threads=1)
    q = r["quality"]
    ref = _reference_predictions(x)
    assert q["full_pass"] and q["samples_predicted"] == N and q["task"] == "regression"
    assert q["rmse"] == pytest.approx(float(np.sqrt(np.mean((ref - y) ** 2))), abs=1e-5)
    assert sorted(lv["label"] for lv in q["per_level"]) == pytest.approx([0.3, 0.6, 1.0])
    cfg = r["pipeline"]["config"]
    assert cfg["threads"] == 1 and cfg["inputs"]["samples"] == N and cfg["inputs"]["sha256"]
    json.dumps(r)


def test_real_inputs_are_fed_as_given(data):
    """--model min-max normalises SYNTHETIC inputs; partners' windows are
    already scaled and must reach the model unchanged."""
    from edgelens.benchmark.onnx_pipeline import OnnxStagePipeline
    x, _, p = data
    o = OnnxStagePipeline(MODEL, inputs=p["npy"])
    o.capture()
    o.preprocess()
    assert np.array_equal(o._prepped["window"][0], x[0])


def test_partial_pass_says_so(data):
    _, _, p = data
    q = run_benchmark(iterations=5, warmup=0, model_path=MODEL, inputs=p["npy"],
                      labels=p["ynpy"])["quality"]
    assert q["samples_predicted"] == 5 and not q["full_pass"]


def test_labels_without_inputs_and_inputs_with_pipeline_are_refused(data, tmp_path):
    _, _, p = data
    with pytest.raises(RuntimeError, match="--labels needs --inputs"):
        run_benchmark(iterations=2, model_path=MODEL, labels=p["y"])
    script = tmp_path / "pipe.py"
    script.write_text("def build_stage_fns():\n    return {'a': lambda: None}\n")
    with pytest.raises(RuntimeError, match="apply to --model runs"):
        run_benchmark(iterations=2, pipeline_path=str(script), inputs=p["npy"])


def test_threads_are_a_session_option():
    from edgelens.benchmark.onnx_pipeline import session_options
    assert session_options(None) is None
    so = session_options(4)
    assert so.intra_op_num_threads == 4 and so.inter_op_num_threads == 1
    with pytest.raises(RuntimeError, match="--threads must be 1 or more"):
        run_benchmark(iterations=1, model_path=MODEL, threads=-1)


# ---- validate / compare / sweep ----

def test_validate_max_rmse_reads_the_quality_block(data):
    _, _, p = data
    r = run_benchmark(iterations=N, warmup=0, model_path=MODEL, inputs=p["npy"], labels=p["ynpy"])
    rmse = r["quality"]["rmse"]
    ok = validate(r, {"max_rmse": rmse + 0.01})
    bad = validate(r, {"max_rmse": rmse - 0.01})
    assert ok["verdict"] == "PASS" and bad["verdict"] == "FAIL"
    acc = validate(r, {"min_accuracy": 0.9})["checks"][0]
    assert acc["status"] == "NOT_MEASURED" and "integer --labels" in acc["reason"]


def test_max_rmse_without_ground_truth_explains_both_routes():
    row = validate({"latency": {}}, {"max_rmse": 0.2})["checks"][0]
    assert row["status"] == "NOT_MEASURED" and "--inputs and --labels" in row["reason"]


def test_compare_tells_setup_from_data_changes():
    def res(threads, sha):
        return {"pipeline": {"config": {"threads": threads, "inputs": {"sha256": sha}}}}
    d = {x["field"]: x["kind"] for x in requirements_diff(res("default", "a"), res(4, "b"))}
    assert d == {"threads": "setup", "inputs": "data"}
    assert requirements_diff(res(4, "a"), res(4, "a")) == []


def test_sweep_reports_accuracy_per_backend(data):
    from edgelens.sweep import run_sweep
    x, y, p = data
    summary, results = run_sweep(MODEL, ["cpu"], iterations=20, inputs_path=p["npy"],
                                 eval_samples=N, labels_path=p["ynpy"], threads=1)
    row = summary["backends"][0]
    ref = _reference_predictions(x)
    assert row["rmse"] == pytest.approx(float(np.sqrt(np.mean((ref - y) ** 2))), abs=1e-5)
    assert results["cpu"]["quality"]["samples"] == N and summary["threads"] == 1
