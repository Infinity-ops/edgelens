"""
edgelens.sweep
----------------
"Which backend?" — answered with speed, energy AND whether the answer
changed. One model, several ONNX Runtime backends (CPU, CUDA, TensorRT FP32
/ FP16 / INT8, DLA) plus external runtimes (EDDL, a C++ engine), all fed
the SAME inputs:

    edgelens sweep --model lstm.onnx --backends cpu,cuda,tensorrt,tensorrt-fp16

Per backend: latency (p50/p99), energy per inference, session start-up
time (TensorRT engine build), and output equivalence against a reference
backend (default cpu):
  max |error|, mean |error|, fraction of values within tolerance, and how
  often the DECISION agrees (argmax for class outputs, a threshold for
  scores such as an anomaly probability).

Decisions: argmax for multi-value float outputs, "> threshold" for single
scores in [0, 1] (or whenever --threshold is given), exact equality for
integer / boolean outputs (class ids, labels). NaN / inf in a backend's
output where the reference is finite counts as an infinite error.

Verdicts:
  EQUIVALENT        every value within tolerance and every decision equal
  NUMERIC_DRIFT     values differ beyond tolerance, decisions still agree
  DECISIONS_DIFFER  some decisions flipped (below --min-agreement)

Speed without equivalence is meaningless: a TensorRT FP16 engine that is
2x faster but flips 3% of anomaly decisions is a different model.

External runtimes: export the inputs (--export-inputs inputs.npz), run them
through e.g. EDDL, save the outputs as .npz (one array per output, first
dimension = sample), and add them with --external eddl=eddl_out.npz
(optionally ,timing=eddl_record.json from `edgelens record` with markers).
"""

import json
import os
import time
from pathlib import Path

import numpy as np

EQUIVALENT, NUMERIC_DRIFT, DECISIONS_DIFFER = "EQUIVALENT", "NUMERIC_DRIFT", "DECISIONS_DIFFER"

TRT = "TensorrtExecutionProvider"
BACKENDS = {
    "cpu": ("CPUExecutionProvider", {}),
    "cuda": ("CUDAExecutionProvider", {}),
    "tensorrt": (TRT, {}),
    "tensorrt-fp16": (TRT, {"trt_fp16_enable": True}),
    # INT8 needs a quantized (QDQ) model or a calibration table; without one
    # TensorRT refuses, and the row says so.
    "tensorrt-int8": (TRT, {"trt_int8_enable": True}),
    # Deep Learning Accelerator (AGX Orin / Xavier): FP16, GPU fallback for
    # layers the DLA can't run.
    "dla": (TRT, {"trt_dla_enable": True, "trt_dla_core": 0, "trt_fp16_enable": True}),
}


def default_trt_cache():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "edgelens", "trt_engines")


def dla_engines():
    """Number of DLA engines on this board (0 on a Jetson Nano, a TX2, a PC)."""
    import glob
    found = set()
    for pattern in ("/dev/nvhost-ctrl-nvdla*", "/dev/nvhost-nvdla*",
                    "/sys/devices/platform/*nvdla*", "/sys/devices/platform/bus@0/*nvdla*",
                    "/sys/devices/platform/host1x/*nvdla*"):
        for p in glob.glob(pattern):
            name = p.rsplit("/", 1)[-1]
            found.add(name.replace("nvhost-ctrl-", "").replace("nvhost-", ""))
    return len(found)


def backend_spec(name, trt_cache=None):
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Known: {', '.join(BACKENDS)}")
    provider, opts = BACKENDS[name]
    opts = dict(opts)
    if provider == TRT and trt_cache:
        os.makedirs(trt_cache, exist_ok=True)
        opts.update(trt_engine_cache_enable=True, trt_engine_cache_path=trt_cache)
    return provider, opts


# ---------------------------------------------------------------- inputs

def synthetic_inputs(onnx_pipe, n, seed=1234):
    from .benchmark.onnx_pipeline import _synthetic_tensor
    rng = np.random.default_rng(seed)
    return [{name: _synthetic_tensor(shape, dtype, rng) for name, shape, dtype in onnx_pipe.inputs}
            for _ in range(n)]


def load_inputs(path, onnx_pipe):
    """.npz with one array per model input (first dim = sample), or .npy for a
    single-input model. A single example (exact input shape) is accepted too."""
    data = np.load(path, allow_pickle=False)
    arrays = {}
    if isinstance(data, np.ndarray):
        if len(onnx_pipe.inputs) != 1:
            raise ValueError(f"{path} is one array but the model has {len(onnx_pipe.inputs)} "
                             f"inputs; use an .npz with one array per input name.")
        arrays[onnx_pipe.inputs[0][0]] = data
    else:
        keys = list(data.keys())
        for name, _, _ in onnx_pipe.inputs:
            if name in data:
                arrays[name] = data[name]
        if not arrays and len(keys) == 1 and len(onnx_pipe.inputs) == 1:
            arrays[onnx_pipe.inputs[0][0]] = data[keys[0]]
        missing = [n for n, _, _ in onnx_pipe.inputs if n not in arrays]
        if missing:
            raise ValueError(f"{path} has no array for input(s) {missing}; found {keys}.")
    n = None
    out = {}
    for name, shape, dtype in onnx_pipe.inputs:
        a = np.asarray(arrays[name])
        if tuple(a.shape) == tuple(shape):
            a = a[None]
        if tuple(a.shape[1:]) != tuple(shape):
            raise ValueError(f"Input '{name}': got {tuple(a.shape)}, expected [N, "
                             f"{', '.join(map(str, shape))}].")
        n = a.shape[0] if n is None else min(n, a.shape[0])
        out[name] = a.astype(dtype, copy=False)
    return [{name: out[name][i] for name in out} for i in range(n)]


def export_inputs(path, samples):
    names = list(samples[0])
    np.savez(path, **{name: np.stack([s[name] for s in samples]) for name in names})
    return path


# ---------------------------------------------------------------- outputs

def run_outputs(session, output_names, samples):
    """{output_name: array [n, ...]} for every sample, in order."""
    per = {name: [] for name in output_names}
    for feed in samples:
        for name, val in zip(output_names, session.run(output_names, feed)):
            per[name].append(np.asarray(val))
    return {name: np.stack(v) for name, v in per.items()}


def load_external_outputs(path, output_names, n):
    data = np.load(path, allow_pickle=False)
    out = {}
    if isinstance(data, np.ndarray):
        if len(output_names) != 1:
            raise ValueError(f"{path}: one array, but the model has {len(output_names)} outputs")
        out[output_names[0]] = data
    else:
        keys = list(data.keys())
        for name in output_names:
            if name in data:
                out[name] = data[name]
        if not out and len(keys) == len(output_names):
            for name, key in zip(output_names, keys):
                out[name] = data[key]
        missing = [n for n in output_names if n not in out]
        if missing:
            raise ValueError(f"{path} has no array for output(s) {missing}; found {keys}")
    for name, a in out.items():
        if a.shape[0] < n:
            raise ValueError(f"{path}: output '{name}' has {a.shape[0]} samples, need {n} "
                             f"(the exported inputs)")
        out[name] = np.asarray(a[:n])
    return out


def _decisions(arr, mode, threshold):
    """(decisions per sample, how). Integer/bool outputs ARE decisions (class
    ids, labels): compared by exact equality."""
    a = np.asarray(arr)
    n = a.shape[0]
    if mode == "auto" and (np.issubdtype(a.dtype, np.integer) or a.dtype == np.bool_):
        mode = "equal"
    if mode == "equal":
        flat = a.reshape(n, -1)
        return [tuple(row) for row in flat.tolist()], "equal"
    flat = a.astype(np.float64).reshape(n, -1)
    if mode == "auto":
        if threshold is not None:
            mode = "threshold"                 # an explicit threshold always wins
        elif flat.shape[1] > 1:
            mode = "argmax"
        elif np.isfinite(flat).all() and flat.min() >= 0.0 and flat.max() <= 1.0:
            mode = "threshold"
        else:
            mode = "none"
    if mode == "argmax":
        return np.argmax(np.nan_to_num(flat, nan=-np.inf), axis=1), "argmax"
    if mode == "threshold":
        t = 0.5 if threshold is None else threshold
        dec = flat > t                          # NaN > t is False: a NaN never "agrees" by luck
        dec = np.where(np.isnan(flat), -1, dec.astype(int))
        return [tuple(row) for row in dec.tolist()], f"> {t:g}"
    return None, "none"


def compare_outputs(ref, cand, atol=1e-5, rtol=1e-4, decision="auto", threshold=None,
                    min_agreement=1.0):
    """Equivalence of a candidate backend's outputs against the reference."""
    per = {}
    worst_abs, agreements, flips, total = 0.0, [], 0, 0
    all_close = True
    for name, r in ref.items():
        c = cand[name]
        if c.shape != r.shape:
            per[name] = {"error": f"shape {tuple(c.shape)} vs reference {tuple(r.shape)}"}
            all_close = False
            continue
        r64, c64 = r.astype(np.float64), c.astype(np.float64)
        # NaN / inf where the reference differs is the worst error there is
        # (FP16 overflow): never let it disappear from max() (NaN compares False).
        same = (c64 == r64) | (np.isnan(c64) & np.isnan(r64))      # also inf == inf
        with np.errstate(invalid="ignore", over="ignore"):
            d = np.where(same, 0.0, np.abs(c64 - r64))
        bad = ~np.isfinite(d)
        close = np.isclose(c64, r64, rtol=rtol, atol=atol, equal_nan=True)
        finite = d[np.isfinite(d)]
        row = {
            "max_abs_error": (float("inf") if bad.any()
                              else (float(finite.max()) if finite.size else 0.0)),
            "mean_abs_error": float(finite.mean()) if finite.size else 0.0,
            "max_rel_error": (float("inf") if bad.any() else
                              float((finite / (np.abs(r64[np.isfinite(d)]) + atol)).max())
                              if finite.size else 0.0),
            "non_finite_values": int(bad.sum()),
            "within_tolerance": float(close.mean()) if close.size else 1.0,
            "samples": int(r.shape[0]),
        }
        dr, how = _decisions(r, decision, threshold)
        if dr is not None:
            mode = {"argmax": "argmax", "equal": "equal"}.get(how, "threshold")
            dc, _ = _decisions(c, mode, threshold if mode == "threshold" else None)
            agree = np.array([a == b for a, b in zip(dr, dc)]) if isinstance(dr, list) \
                else (np.asarray(dr) == np.asarray(dc))
            row.update(decision=how, decision_agreement=float(agree.mean()),
                       decisions_flipped=int((~agree).sum()))
            agreements.append(row["decision_agreement"])
            flips += row["decisions_flipped"]
            total += int(agree.size)
        else:
            row["decision"] = "none"
        worst_abs = max(worst_abs, row["max_abs_error"])
        all_close = all_close and row["within_tolerance"] == 1.0
        per[name] = row
    agreement = min(agreements) if agreements else None
    if agreement is not None and agreement < min_agreement:
        verdict = DECISIONS_DIFFER
    elif all_close:
        verdict = EQUIVALENT
    else:
        verdict = NUMERIC_DRIFT
    non_finite = sum(r.get("non_finite_values", 0) for r in per.values())
    return {"verdict": verdict, "max_abs_error": worst_abs, "non_finite_values": non_finite,
            "decision_agreement": agreement,
            "decisions_flipped": flips, "decisions_compared": total,
            "tolerance": {"atol": atol, "rtol": rtol}, "min_agreement": min_agreement,
            "outputs": per}


# ---------------------------------------------------------------- the sweep

def run_sweep(model, backends, reference="cpu", iterations=300, input_shapes=None,
              inputs_path=None, eval_samples=50, export_inputs_path=None, external=None,
              atol=1e-5, rtol=1e-4, decision="auto", threshold=None, min_agreement=1.0,
              deadline_ms=None, trt_cache=None, save_stem=None, progress=None):
    """Returns (summary, {backend: result_document})."""
    from .benchmark.onnx_pipeline import OnnxStagePipeline
    from .core.harness import run_harness
    from .core.pipeline import Pipeline

    say = progress or (lambda msg: None)
    order = [reference] + [b for b in backends if b != reference]
    rows, results = [], {}
    ref_out, samples, output_names, inputs_source = None, None, None, None

    for name in order:
        provider, opts = backend_spec(name, trt_cache)
        row = {"backend": name, "provider": provider, "options": opts}
        if opts.get("trt_dla_enable") and not dla_engines():
            # TensorRT falls back to the GPU without a word when there is no DLA
            # (fourth Nano run: a "dla" row with GPU numbers) — that is not a DLA result
            row.update(status="unavailable", reason="this board has no DLA engine; TensorRT "
                       "would silently run the model on the GPU (DLA: Xavier / Orin)")
            rows.append(row)
            continue
        say(f"{name}: creating session (TensorRT may build an engine; this can take minutes "
            f"the first time)...")
        t0 = time.perf_counter()
        try:
            onnx = OnnxStagePipeline(model, provider=provider, input_shapes=input_shapes,
                                     provider_options=opts or None)
        except RuntimeError as e:
            row.update(status="unavailable", reason=str(e).splitlines()[0][:300])
            rows.append(row)
            if name == reference:
                raise RuntimeError(f"The reference backend '{reference}' is not usable here: "
                                   f"{row['reason']}") from e
            continue
        row["session_init_s"] = round(time.perf_counter() - t0, 3)
        if samples is None:
            if inputs_path:
                samples = load_inputs(inputs_path, onnx)[:eval_samples or None]
                inputs_source = f"file:{inputs_path}"
            else:
                samples = synthetic_inputs(onnx, eval_samples)
                inputs_source = "synthetic (seeded random)"
            output_names = onnx.output_names
            if export_inputs_path:
                export_inputs(export_inputs_path, samples)
        try:
            say(f"{name}: computing outputs for {len(samples)} input samples...")
            outputs = run_outputs(onnx.session, output_names, samples)
        except Exception as e:                                   # a backend that can't run
            if name == reference:
                raise RuntimeError(f"The reference backend '{reference}' could not run the "
                                   f"model: {e}"[:400]) from e
            row.update(status="error", reason=f"inference failed: {e}"[:300])
            rows.append(row)
            continue
        if name == reference:
            ref_out = outputs
            eq = compare_outputs(ref_out, outputs, atol, rtol, decision, threshold, min_agreement)
            eq["reference"] = True
        else:
            eq = compare_outputs(ref_out, outputs, atol, rtol, decision, threshold, min_agreement)
        eq.update(reference_backend=reference, inputs=inputs_source, samples=len(samples))

        say(f"{name}: timing {iterations} iterations...")
        pipe = Pipeline.from_stage_fns(onnx.stage_fns(), name=f"onnx-{name}", pack="vision",
                                       model_inputs=onnx.input_summary(),
                                       provider=onnx.provider, provider_options=opts,
                                       provider_fallbacks=onnx.provider_fallbacks)
        res = run_harness(pipe, iterations=iterations, warmup=10, deadline_ms=deadline_ms,
                          pipeline_source=f"onnxruntime:{name} ({model})", model_path=model)
        res["backend"] = {"name": name, "provider": provider, "options": opts,
                          "session_init_s": row["session_init_s"]}
        res["equivalence"] = eq
        results[name] = res
        rows.append(_summary_row(row, res, eq))
        if save_stem:
            _write(f"{save_stem}.{name}.json", res)

    for spec in external or []:
        rows.append(_external_row(spec, ref_out, output_names, samples, reference, atol, rtol,
                                  decision, threshold, min_agreement, inputs_source))

    summary = {"kind": "edgelens-sweep", "model": str(model), "reference": reference,
               "inputs": inputs_source, "samples": len(samples) if samples else 0,
               "iterations": iterations,
               "tolerance": {"atol": atol, "rtol": rtol}, "decision": decision,
               "threshold": threshold, "min_agreement": min_agreement, "backends": rows}
    if save_stem:
        _write(f"{save_stem}.json", summary)
    return summary, results


def _summary_row(row, res, eq):
    lat = res.get("latency") or {}
    stats = res.get("stage_stats_ms") or {}
    en = res.get("energy") or {}
    row.update(
        status="ok",
        p50_ms=lat.get("p50"), p99_ms=lat.get("p99"), mean_ms=lat.get("mean"),
        inference_p50_ms=(stats.get("inference") or {}).get("p50"),
        energy_mj=(round(en["energy_per_iteration_j"] * 1000, 3)
                   if en.get("available") and not en.get("low_confidence") else None),
        verdict=eq["verdict"], max_abs_error=eq["max_abs_error"],
        decision_agreement=eq["decision_agreement"], decisions_flipped=eq["decisions_flipped"],
        run_id=(res.get("identity") or {}).get("run_id"))
    return row


def _external_row(spec, ref_out, output_names, samples, reference, atol, rtol, decision,
                  threshold, min_agreement, inputs_source):
    name, outputs_path, timing_path = spec
    row = {"backend": name, "provider": "external", "options": {}}
    try:
        cand = load_external_outputs(outputs_path, output_names, len(samples))
        eq = compare_outputs(ref_out, cand, atol, rtol, decision, threshold, min_agreement)
    except (OSError, ValueError, KeyError) as e:
        row.update(status="error", reason=str(e)[:300])
        return row
    row.update(status="ok", verdict=eq["verdict"], max_abs_error=eq["max_abs_error"],
               decision_agreement=eq["decision_agreement"],
               decisions_flipped=eq["decisions_flipped"], equivalence=eq,
               outputs_file=str(outputs_path))
    if timing_path:
        try:
            t = json.loads(Path(timing_path).read_text())
            lat = t.get("latency") or {}
            en = t.get("energy") or {}
            row.update(p50_ms=lat.get("p50"), p99_ms=lat.get("p99"), mean_ms=lat.get("mean"),
                       energy_mj=(round(en["energy_per_iteration_j"] * 1000, 3)
                                  if en.get("available") and not en.get("low_confidence")
                                  else None),
                       timing_file=str(timing_path),
                       timing_source=t.get("mode"))
        except (OSError, ValueError) as e:
            row["timing_error"] = str(e)[:200]
    return row


def parse_external(text):
    """'eddl=out.npz' or 'eddl=out.npz,timing=run.json' -> (name, outputs, timing)."""
    if "=" not in text:
        raise ValueError(f"--external expects NAME=outputs.npz[,timing=result.json], got '{text}'")
    name, rest = text.split("=", 1)
    parts = rest.split(",")
    outputs, timing = parts[0], None
    for p in parts[1:]:
        if p.startswith("timing="):
            timing = p[len("timing="):]
    return name.strip(), outputs, timing


def _write(path, doc):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=2, default=_np_default), encoding="utf-8")


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)
