"""
edgelens.benchmark.runner
---------------------------
`run_benchmark()` — the entry point behind `edgelens benchmark`. Since
v0.1.0 it is a thin adapter: every source of work (an .onnx model, a
--pipeline script, a stage_fns dict, or an edgelens.Pipeline) becomes an
edgelens.Pipeline and runs on the same generic engine
(edgelens.core.harness). Demo mode builds the same result document from
synthetic timings.

Real hardware mode requires something real to measure: there is no silent
placeholder pipeline. A benchmark you didn't wire a model or stages into
raises RuntimeError instead of reporting timings of nothing.
"""

from ..core.harness import StageError
from ..core.identity import attach_identity
from ..core.pipeline import Pipeline
from ..core.result import build_result
from ..demo import simulator
from ..hardware import detector
from ..packs import get_pack
from ..packs.vision import VISION_STAGE_NAMES

# Backward-compatible constant: the vision pack's classic stage template.
STAGES = list(VISION_STAGE_NAMES)


def run_benchmark(iterations=50, warmup=10, demo=False, demo_scenario="balanced",
                  model_path=None, provider=None, stage_fns=None, pipeline_path=None,
                  telemetry_interval_s=0.1, input_shapes=None, pipeline=None,
                  deadline_ms=None, period_ms=None, pace=False, idle_baseline_s=None,
                  source=None):
    """Run a benchmark and return a result document
    (see edgelens.core.result for the shape).

    Exactly one pipeline source in hardware mode: model_path, pipeline_path,
    stage_fns (dict, any stage names) or pipeline (edgelens.Pipeline).

    Raises:
        RuntimeError: no pipeline source in hardware mode, more than one
            source, or a stage failing (message names the stage).
    """
    given = [n for n, v in (("stage_fns", stage_fns), ("pipeline_path", pipeline_path),
                            ("model_path", model_path), ("pipeline", pipeline)) if v is not None]
    if demo:
        use_demo = True
    elif given:
        use_demo = False
    else:
        # A bare `edgelens benchmark` on a laptop still previews in demo mode;
        # on a real Jetson with nothing wired in, fail loudly instead.
        use_demo = not detector.is_jetson()

    if use_demo:
        result = _demo_result(iterations, demo_scenario, deadline_ms)
        attach_identity(result)
        return result

    pipe, source_label, model = _resolve_pipeline(given, model_path, provider, stage_fns,
                                                  pipeline_path, input_shapes, pipeline)
    try:
        return pipe.run(iterations=iterations, warmup=warmup, source=source,
                        deadline_ms=deadline_ms, period_ms=period_ms, pace=pace,
                        telemetry_interval_s=telemetry_interval_s,
                        idle_baseline_s=idle_baseline_s, pipeline_source=source_label)
    except StageError as e:
        raise RuntimeError(str(e)) from e
    except ValueError as e:
        raise RuntimeError(str(e)) from e


def _resolve_pipeline(given, model_path, provider, stage_fns, pipeline_path, input_shapes,
                      pipeline):
    if len(given) > 1:
        raise RuntimeError(
            f"Ambiguous pipeline source: got {', '.join(given)} together. "
            f"Pass exactly one of --model, --pipeline, or stage_fns=... / pipeline=... "
            f"from Python."
        )
    if pipeline is not None:
        return pipeline, f"pipeline:{pipeline.name}", None
    if stage_fns is not None:
        try:
            return Pipeline.from_stage_fns(stage_fns, name="user-supplied"), "user-supplied", None
        except (TypeError, ValueError) as e:
            raise RuntimeError(str(e)) from e
    if pipeline_path is not None:
        from .pipeline_loader import load_pipeline_from_script
        pipe = load_pipeline_from_script(pipeline_path)
        return pipe, f"custom pipeline ({pipeline_path})", None
    if model_path is not None:
        from .onnx_pipeline import OnnxStagePipeline
        onnx = OnnxStagePipeline(model_path, provider=provider, input_shapes=input_shapes)
        pipe = Pipeline.from_stage_fns(onnx.stage_fns(), name="onnx", pack="vision",
                                       model_inputs=onnx.input_summary(),
                                       provider=onnx.provider)
        # identity needs the model hash: run() attaches identity, re-attach with it below
        pipe._model_path = model_path
        return pipe, f"onnxruntime:{onnx.provider} ({model_path})", model_path
    raise RuntimeError(
        "edgelens benchmark needs something real to measure.\n"
        "Pass one of:\n"
        "  --model path/to/model.onnx       (runs a real ONNX Runtime session)\n"
        "  --pipeline path/to/script.py      (your own capture/model/pipeline)\n"
        "  --demo                            (synthetic data for previewing output)\n"
        "  stage_fns=... / pipeline=... from Python\n"
        "EdgeLens does not fabricate hardware-mode results from a placeholder "
        "pipeline — see ROADMAP.md."
    )


def _demo_result(iterations, scenario, deadline_ms):
    pack = get_pack("vision")
    samples = {s: [] for s in STAGES}
    e2e = []
    for _ in range(iterations):
        t = simulator.simulate_stage_timings(bottleneck=scenario)
        for s in STAGES:
            samples[s].append(t[s])
        e2e.append(sum(t[s] for s in STAGES))
    desc = {"name": f"demo-{scenario}", "pack": "vision", "config": {"demo_scenario": scenario},
            "stages": [{"name": s, "role": pack.role_for(s)} for s in STAGES]}
    tel = simulator.simulate_telemetry(bottleneck=scenario)
    return build_result(pipeline_desc=desc, stage_samples_ms=samples, e2e_ms=e2e,
                        mode="demo", pipeline_source="simulated", telemetry_summary=tel,
                        deadline_ms=deadline_ms, pack=pack)
