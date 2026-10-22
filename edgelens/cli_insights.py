"""
edgelens.cli_insights
-----------------------
v0.2 Layer 2 commands — insights nobody else gives you:

    edgelens sweep     which backend? speed, energy AND output equivalence
    edgelens periodic  latency and energy at the real inference rate (1 Hz...)
    edgelens soak      30-minute+ runs: drift, throttling, memory growth

Registered on the main app by edgelens.cli.
"""

import json
from pathlib import Path
from typing import List

import typer
from rich.panel import Panel
from rich.table import Table

from .benchmark.onnx_pipeline import parse_input_shapes
from .cli import (INPUTS_HELP, LABELS_HELP, ORDER_HELP, THREADS_HELP, _exit_on_write_error, _ms,
                  _print_integrity_issues, _print_quality, _write_or_exit, app, console)


def _fmt(v, nd=3, unit=""):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}{unit}" if abs(v) < 1e-3 and v != 0 else f"{round(v, nd)}{unit}"
    return f"{v}{unit}"


def _stem(save):
    p = Path(save)
    return str(p.with_suffix("")) if p.suffix == ".json" else str(p)


# ---------------------------------------------------------------- sweep (I3)

@app.command()
def sweep(
    model: str = typer.Option(..., "--model", help="ONNX model to sweep."),
    backends: str = typer.Option("cpu,cuda,tensorrt,tensorrt-fp16", "--backends",
                                 help="Comma list: cpu, cuda, tensorrt, tensorrt-fp16, "
                                      "tensorrt-int8, dla. Unavailable ones are reported, "
                                      "not fatal."),
    reference: str = typer.Option("cpu", "--reference", help="Backend whose outputs are the "
                                  "reference for equivalence."),
    iterations: int = typer.Option(300, "--iterations", min=1, help="Timed iterations per "
                                   "backend."),
    inputs: str = typer.Option(None, "--inputs", help="Real input data: .npz (one array per "
                               "model input, first dim = sample) or .npy. Default: seeded "
                               "synthetic inputs (weaker evidence for decisions)."),
    eval_samples: int = typer.Option(50, "--eval-samples", min=1, help="Input samples used "
                                     "for output equivalence."),
    export_inputs: str = typer.Option(None, "--export-inputs", help="Write the evaluation "
                                      "inputs to this .npz, to run them through an external "
                                      "runtime (EDDL, C++)."),
    external: List[str] = typer.Option(None, "--external", help="NAME=outputs.npz[,timing="
                                       "result.json]: outputs of an external runtime for the "
                                       "exported inputs. Repeatable."),
    atol: float = typer.Option(1e-5, "--atol", help="Absolute tolerance for 'equivalent'."),
    rtol: float = typer.Option(1e-4, "--rtol", help="Relative tolerance for 'equivalent'."),
    decision: str = typer.Option("auto", "--decision", help="How outputs become decisions: "
                                 "auto | argmax | threshold | none."),
    threshold: float = typer.Option(None, "--threshold", help="Decision threshold for score "
                                    "outputs (default 0.5 for outputs in [0, 1])."),
    min_agreement: float = typer.Option(1.0, "--min-agreement", help="Decision agreement "
                                        "below this = DECISIONS_DIFFER."),
    input_shape: List[str] = typer.Option(None, "--input-shape", help="Shape for dynamic "
                                          "inputs, e.g. window:1x50x16."),
    deadline_ms: float = typer.Option(None, "--deadline-ms", help="Per-inference deadline."),
    trt_cache: str = typer.Option(None, "--trt-cache", help="TensorRT engine cache folder "
                                  "(default ~/.cache/edgelens/trt_engines)."),
    no_trt_cache: bool = typer.Option(False, "--no-trt-cache", help="Rebuild TensorRT "
                                      "engines every run."),
    save: str = typer.Option("edgelens_sweep.json", "--save", help="Summary JSON; each "
                             "backend's full result goes next to it as NAME.<backend>.json."),
    labels: str = typer.Option(None, "--labels", help="With --inputs: one ground-truth value "
                               "per sample; adds each backend's RMSE (or accuracy)."),
    threads: int = typer.Option(None, "--threads", help=THREADS_HELP),
):
    """Which backend? Latency, energy AND whether the answer changed, for the
    same model on CPU / CUDA / TensorRT (FP32, FP16, INT8) / DLA and external
    runtimes, all fed the same inputs."""
    from . import sweep as sw
    names = [b.strip() for b in backends.split(",") if b.strip()]
    try:
        for b in names + [reference]:
            sw.backend_spec(b)
        ext = [sw.parse_external(e) for e in (external or [])]
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2)
    if ext and not export_inputs and not inputs:
        console.print("[yellow]--external outputs must come from the SAME inputs: export them "
                      "first with --export-inputs (or pass the same --inputs file).[/yellow]")
    try:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _exit_on_write_error(save, e)
    cache = None if no_trt_cache else (trt_cache or sw.default_trt_cache())
    try:
        with console.status("Sweeping backends...") as status:
            summary, results = sw.run_sweep(
                model, names, reference=reference, iterations=iterations,
                input_shapes=parse_input_shapes(input_shape), inputs_path=inputs,
                eval_samples=eval_samples, export_inputs_path=export_inputs, external=ext,
                atol=atol, rtol=rtol, decision=decision, threshold=threshold,
                min_agreement=min_agreement, deadline_ms=deadline_ms, trt_cache=cache,
                save_stem=_stem(save), progress=lambda m: status.update(m), threads=threads,
                labels_path=labels)
    except (RuntimeError, ValueError, OSError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    print_sweep(summary)
    if export_inputs:
        console.print(f"[dim]Evaluation inputs written to {export_inputs} (run them through "
                      f"your external runtime, save the outputs as .npz, add with "
                      f"--external NAME=out.npz).[/dim]")
    console.print(f"[green]Saved →[/green] {save} [dim](+ one result per backend: "
                  f"{_stem(save)}.<backend>.json — use them with validate / compare)[/dim]")


def print_sweep(summary):
    t = Table(title=f"EdgeLens sweep — {Path(summary['model']).name} "
                    f"(reference: {summary['reference']}, {summary['samples']} input samples, "
                    f"{summary['inputs']})")
    acc = any(r.get("rmse") is not None or r.get("accuracy") is not None
              for r in summary["backends"])
    cols = ["Backend", "Start-up", "p50", "p99", "Energy/inf", "Max |error|",
            "Decisions agree", "Verdict"] + (["vs labels"] if acc else [])
    for col in cols:
        t.add_column(col)
    style = {"EQUIVALENT": "green", "NUMERIC_DRIFT": "yellow", "DECISIONS_DIFFER": "red"}
    for r in summary["backends"]:
        if r.get("status") != "ok":
            t.add_row(r["backend"], "-", "-", "-", "-", "-", "-",
                      f"[dim]{r.get('status')}: {r.get('reason', '')[:70]}[/dim]",
                      *(["-"] if acc else []))
            continue
        agree = r.get("decision_agreement")
        agree_s = "-" if agree is None else (
            f"{agree * 100:.2f}%" + (f" ({r['decisions_flipped']} flipped)"
                                     if r.get("decisions_flipped") else ""))
        v = r.get("verdict")
        t.add_row(r["backend"], _fmt(r.get("session_init_s"), 3, " s"),
                  _ms(r.get("p50_ms")), _ms(r.get("p99_ms")),
                  _fmt(r.get("energy_mj"), 3, " mJ"),
                  "0" if r.get("max_abs_error") == 0 else f"{r.get('max_abs_error', 0):.3g}",
                  agree_s, f"[{style.get(v, 'white')}]{v}[/{style.get(v, 'white')}]",
                  *([f"RMSE {r['rmse']}" if r.get("rmse") is not None else
                     f"{r['accuracy'] * 100:.2f}%" if r.get("accuracy") is not None else "-"]
                    if acc else []))
    console.print(t)
    tol = summary["tolerance"]
    console.print(f"[dim]EQUIVALENT = every value within atol {tol['atol']:g} / rtol "
                  f"{tol['rtol']:g} and every decision equal. NUMERIC_DRIFT = values differ, "
                  f"decisions agree. DECISIONS_DIFFER = decisions flipped. Times in ms; "
                  f"start-up includes TensorRT engine builds.[/dim]")
    if summary["inputs"].startswith("synthetic"):
        console.print("[yellow]Inputs were synthetic: decision agreement on random data says "
                      "little about real decisions. Pass --inputs with recorded data (e.g. the "
                      "merged DEM2 windows) for evidence that counts.[/yellow]")


# ---------------------------------------------------------------- periodic (I1)

def _build_pipe(model, pipeline, provider, input_shape, inputs=None, labels=None, threads=None,
                inputs_order=None):
    from .benchmark.runner import _resolve_pipeline
    given = [n for n, v in (("pipeline_path", pipeline), ("model_path", model)) if v]
    if len(given) != 1:
        console.print("[red]Give exactly one of --model or --pipeline.[/red]")
        raise typer.Exit(2)
    try:
        return _resolve_pipeline(given, model, provider, None, pipeline,
                                 parse_input_shapes(input_shape), None, inputs=inputs,
                                 labels=labels, threads=threads, inputs_order=inputs_order)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command()
def periodic(
    model: str = typer.Option(None, "--model", help="ONNX model."),
    pipeline: str = typer.Option(None, "--pipeline", help="Pipeline script (build_pipeline)."),
    provider: str = typer.Option(None, "--provider", help="ONNX Runtime provider (--model)."),
    input_shape: List[str] = typer.Option(None, "--input-shape", help="Dynamic input shape."),
    period_ms: List[float] = typer.Option(None, "--period-ms", help="Inference period; "
                                          "repeat for a sweep, e.g. --period-ms 10 "
                                          "--period-ms 100 --period-ms 1000. Default 1000."),
    iterations: int = typer.Option(30, "--iterations", min=2, help="Inferences per period "
                                   "(>= 30 for trustworthy energy)."),
    warm_iterations: int = typer.Option(200, "--warm-iterations", min=10,
                                        help="Back-to-back iterations for the comparison."),
    idle_baseline: float = typer.Option(10.0, "--idle-baseline", help="Seconds of idle power "
                                        "measurement first (needs a power sensor)."),
    deadline_ms: float = typer.Option(None, "--deadline-ms", help="Deadline (default: the "
                                      "period)."),
    save: str = typer.Option("edgelens_periodic.json", "--save", help="Output JSON path."),
    inputs: str = typer.Option(None, "--inputs", help=INPUTS_HELP),
    labels: str = typer.Option(None, "--labels", help=LABELS_HELP),
    threads: int = typer.Option(None, "--threads", help=THREADS_HELP),
    inputs_order: str = typer.Option("sequential", "--inputs-order", help=ORDER_HELP),
):
    """Latency and energy at the REAL inference rate. Benchmarks run back to
    back; a 1 Hz model wakes a sleeping board every second and pays for it."""
    from . import periodic as per
    periods = list(period_ms or [1000.0])
    pipe, source, model_path = _build_pipe(model, pipeline, provider, input_shape, inputs, labels,
                                           threads, inputs_order)
    total = sum(periods) * (iterations + 1) / 1000.0
    console.print(f"[dim]This takes about {total + idle_baseline + 5:.0f} s "
                  f"({len(periods)} period(s) x {iterations} inferences).[/dim]")
    try:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _exit_on_write_error(save, e)
    try:
        with console.status("Measuring...") as status:
            result = per.run_periodic(pipe, periods, iterations=iterations,
                                      warm_iterations=warm_iterations,
                                      idle_baseline_s=idle_baseline, deadline_ms=deadline_ms,
                                      pipeline_source=source, model_path=model_path,
                                      progress=lambda m: status.update(m))
    except (RuntimeError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    from .benchmark.runner import attach_quality
    attach_quality(result, pipe)
    print_periodic(result)
    _print_quality(result.get("quality"))
    _write_or_exit(Path(save), json.dumps(result, indent=2))
    console.print(f"[green]Saved →[/green] {save}")


def print_periodic(result):
    p = result["periodic"]
    b = p["back_to_back"]
    t = Table(title="EdgeLens periodic — latency and energy at the real rate")
    for col in ("Rate", "p50", "p99", "vs back-to-back", "OS wake-up p50", "Energy / period",
                "Extra / inference", "Avg power"):
        t.add_column(col)
    t.add_row("back to back", _ms(b["p50_ms"]), _ms(b["p99_ms"]), "1.00x", "-",
              "-", _fmt(b.get("energy_per_inference_mj"), 3, " mJ"),
              _fmt(b.get("power_w_mean"), 3, " W"))
    for r in p["periods"]:
        hz = 1000.0 / r["period_ms"]
        ratio = r.get("penalty_ratio_p50")
        rs = "-" if ratio is None else f"{ratio:.2f}x"
        if ratio and ratio >= 1.5:
            rs = f"[bold red]{rs}[/bold red]"
        t.add_row(f"every {r['period_ms']:g} ms ({hz:g} Hz)", _ms(r["p50_ms"]), _ms(r["p99_ms"]),
                  rs, _ms(r.get("wake_delay_p50_ms")), _fmt(r.get("energy_per_period_mj"), 4, " mJ"),
                  (f"< {r['dynamic_energy_below_mj']:g} mJ (below sensor resolution)"
                   if r.get("dynamic_energy_below_mj") is not None
                   else _fmt(r.get("dynamic_energy_per_inference_mj"), 3, " mJ")),
                  _fmt(r.get("power_w_mean"), 3, " W"))
    console.print(t)
    console.print("[dim]p50 / p99 = the pipeline's own work per inference (service time). OS "
                  "wake-up = how late the loop started after its release time — a separate "
                  "cost, not the model's.[/dim]")
    if p.get("idle_power_w") is not None:
        console.print(f"[dim]Idle baseline: {p['idle_power_w']} W. Energy / period includes "
                      f"idle time (what the battery pays); extra / inference is above idle "
                      f"(what the model costs).[/dim]")
    first = p["primary"]
    if p["duty_cycle_penalty"]:
        worst = max(p["periods"], key=lambda r: r.get("penalty_ratio_p50") or 0)
        rail = {True: "GPU power-gating (railgate) is ON", False: "GPU power-gating is off",
                None: "GPU power-gating unknown"}[p.get("gpu_railgate")]
        console.print(Panel(
            f"Your benchmark says [bold]{b['p50_ms']:.3f} ms[/bold]. Every "
            f"{worst['period_ms']:g} ms it's [bold]{worst['p50_ms']:.3f} ms[/bold] "
            f"({worst['penalty_ratio_p50']:.1f}x). Between inferences the board saves power "
            f"(clocks ramp down, caches go cold; {rail}) and every inference pays to wake "
            f"it up. A back-to-back benchmark never shows this.\n"
            f"[cyan]Next: run again with clocks pinned (sudo jetson_clocks) and `edgelens "
            f"compare` both files — it shows the latency gained AND the power it costs.[/cyan]",
            title="DUTY_CYCLE_PENALTY", border_style="yellow"))
    elif first.get("penalty_ratio_p50") is not None:
        console.print(f"[green]No duty-cycle penalty: periodic p50 is "
                      f"{first['penalty_ratio_p50']:.2f}x the back-to-back p50.[/green]")
    for n in p.get("notes") or []:
        console.print(f"[dim]{n}[/dim]")
    for r in p["periods"]:
        if r.get("energy_low_confidence"):
            console.print(f"[yellow]Energy at {r['period_ms']:g} ms: "
                          f"{r['energy_low_confidence']}[/yellow]")


# ---------------------------------------------------------------- soak (I2)

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def soak(
    ctx: typer.Context,
    duration: str = typer.Option(None, "--duration", help="How long: 30m, 2h, 90s, 1h30m."),
    window: float = typer.Option(60.0, "--window", help="Seconds per window (one line in "
                                 "the .jsonl each)."),
    model: str = typer.Option(None, "--model", help="ONNX model (EdgeLens drives the loop)."),
    pipeline: str = typer.Option(None, "--pipeline", help="Pipeline script (EdgeLens drives "
                                 "the loop)."),
    provider: str = typer.Option(None, "--provider", help="ONNX Runtime provider (--model)."),
    input_shape: List[str] = typer.Option(None, "--input-shape", help="Dynamic input shape."),
    period_ms: float = typer.Option(None, "--period-ms", help="Release an iteration every "
                                    "N ms (e.g. 1000 for a 1 Hz model). Default: back to back."),
    pid: int = typer.Option(None, "--pid", help="Observe a running process instead."),
    name: str = typer.Option(None, "--name", help="Observe the process with this name."),
    deadline_ms: float = typer.Option(None, "--deadline-ms", help="Deadline per iteration."),
    report: str = typer.Option(None, "--report", help="Only rebuild the analysis from an "
                               "existing .jsonl (e.g. after a crash)."),
    save: str = typer.Option("edgelens_soak.json", "--save", help="Result JSON; windows "
                             "stream to the same name with .jsonl."),
    inputs: str = typer.Option(None, "--inputs", help=INPUTS_HELP),
    labels: str = typer.Option(None, "--labels", help=LABELS_HELP),
    threads: int = typer.Option(None, "--threads", help=THREADS_HELP),
    inputs_order: str = typer.Option("sequential", "--inputs-order", help=ORDER_HELP),
):
    """Long runs: latency drift, thermal throttling, memory growth (process
    and Jetson GPU memory), swap and time-to-out-of-memory, window by window,
    streamed to disk.   edgelens soak --duration 30m --pipeline dem2.py --period-ms 1000
                        edgelens soak --duration 30m -- ./fortis --config x.yaml"""
    from . import soak as sk
    cmd = list(ctx.args)
    if report:
        try:
            result = sk.report_from_jsonl(report)
        except OSError as e:
            console.print(f"[red]Cannot read {report}: {e}[/red]")
            raise typer.Exit(1)
        print_soak(result)
        _write_or_exit(Path(save), json.dumps(result, indent=2))
        console.print(f"[green]Saved →[/green] {save}")
        return
    if not duration:
        console.print("[red]--duration is required (e.g. --duration 30m).[/red]")
        raise typer.Exit(2)
    try:
        seconds = sk.parse_duration(duration)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2)
    driven = bool(model or pipeline)
    observed = bool(cmd or pid or name)
    if driven == observed:
        console.print("[red]Give either --model/--pipeline (EdgeLens drives the loop) or a "
                      "target to observe (-- CMD, --pid, --name).[/red]")
        raise typer.Exit(2)
    jsonl = str(Path(save).with_suffix(".jsonl"))
    try:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _exit_on_write_error(save, e)
    console.print(f"[dim]Soak for {seconds / 60:.1f} min in {window:g} s windows; every window "
                  f"is appended to {jsonl} as it completes (Ctrl-C stops and keeps the "
                  f"analysis).[/dim]")
    try:
        if driven:
            pipe, source, model_path = _build_pipe(model, pipeline, provider, input_shape,
                                                   inputs, labels, threads, inputs_order)
            result = sk.run_soak_pipeline(pipe, seconds, window_s=window, period_ms=period_ms,
                                          deadline_ms=deadline_ms, jsonl_path=jsonl,
                                          pipeline_source=source, model_path=model_path,
                                          progress=lambda m: console.print(f"[dim]{m}[/dim]"))
        else:
            result = sk.run_soak_observed(seconds, window_s=window, cmd=cmd or None, pid=pid,
                                          name=name, deadline_ms=deadline_ms, jsonl_path=jsonl,
                                          progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    except (RuntimeError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    if driven:
        from .benchmark.runner import attach_quality
        attach_quality(result, pipe)
    print_soak(result)
    _print_quality(result.get("quality"))
    _write_or_exit(Path(save), json.dumps(result, indent=2))
    console.print(f"[green]Saved →[/green] {save}  [dim](windows: {jsonl})[/dim]")


def print_soak(result):
    s = result["soak"]
    t = Table(title=f"EdgeLens soak — {s['windows']} windows, {s['duration_s'] / 60:.1f} min")
    t.add_column("What")
    t.add_column("Result")
    for k in ("p50", "p99"):
        d = s["drift"][k]
        if d.get("change_pct") is None:
            t.add_row(f"Latency {k} drift", f"[dim]{d.get('note') or 'not enough windows (needs 4+)'}[/dim]")
            continue
        verdict = ("[red]DRIFTING[/red]" if d["drifting"] else
                   "[green]stable[/green]" if not d["significant"] else "[yellow]small trend[/yellow]")
        t.add_row(f"Latency {k} drift",
                  f"{d['start']:.3f} → {d['end']:.3f} ms ({d['change_pct']:+.1f}%, "
                  f"{'significant' if d['significant'] else 'not significant'}) {verdict}")
    lv = s.get("latency_levels")
    if lv:
        clock = ""
        if lv.get("cpu_freq_mhz_low_level") and lv.get("cpu_freq_mhz_high_level"):
            clock = (f"; CPU clock {lv['cpu_freq_mhz_low_level']:.0f} MHz in the fast windows, "
                     f"{lv['cpu_freq_mhz_high_level']:.0f} MHz in the slow ones")
        t.add_row("Latency levels", f"[yellow]two: p50 ~{lv['low_p50_ms']} ms in "
                  f"{lv['windows_low']} windows, ~{lv['high_p50_ms']} ms in {lv['windows_high']} "
                  f"({lv['switches']} switches){clock}[/yellow]")
    th = s["throttling"]
    t.add_row("Thermal throttling", f"[red]{th['throttled_windows']} window(s)[/red]"
              if th["throttled_windows"] else "[green]none detected[/green]")
    t.add_row("Peak temperature", f"{s['max_temp_c']} C" if s.get("max_temp_c") else "-")
    m = s["memory"]

    def rate(v, sig):
        return "-" if v is None else f"{v:+.2f} MB/h" + ("" if sig else " (not significant)")
    verdict = ((" [red]LEAK SUSPECTED[/red]" if m.get("leak_suspected") else "")
               + (f" [yellow]growing by {m.get('leak_grown_mb', m.get('process_rss_mb_grown'))} "
                  f"MB over the run — too little to call a leak (warm-up looks the same); run "
                  f"longer[/yellow]" if m.get("rss_growing_inconclusive") else ""))
    if m.get("process_anon_mb_per_h") is not None:
        # the leak verdict is on private memory; RSS also counts mapped files
        t.add_row("Process memory (private)", rate(m.get("process_anon_mb_per_h"),
                                                   m.get("process_anon_mb_per_h_significant"))
                  + verdict)
        t.add_row("Process memory (RSS)", rate(m.get("process_rss_mb_per_h"),
                                               m.get("process_rss_mb_per_h_significant")))
        if m.get("file_backed_growth"):
            t.add_row("Memory-mapped files", f"{m.get('process_file_mb_per_h'):+.2f} MB/h "
                      f"[cyan](+{m.get('process_file_mb_grown')} MB: file pages read for the "
                      f"first time, e.g. a memory-mapped input file — cache, not a leak)[/cyan]")
    else:
        t.add_row("Process memory (RSS)", rate(m.get("process_rss_mb_per_h"),
                                               m.get("process_rss_mb_per_h_significant"))
                  + verdict)
    if m.get("nvmap_used_mb_last") is not None:
        t.add_row("GPU memory (NvMap)", rate(m.get("gpu_nvmap_mb_per_h"),
                                             m.get("gpu_nvmap_mb_per_h_significant"))
                  + (" [red]GROWING[/red]" if m.get("gpu_memory_growth_suspected") else ""))
    t.add_row("System available memory", rate(m.get("available_mb_per_h"),
                                              m.get("available_mb_per_h_significant"))
              + (f" — runs out in ~{m['hours_until_out_of_memory']} h"
                 if m.get("hours_until_out_of_memory") else ""))
    if m.get("swap_grew_mb"):
        t.add_row("Swap", f"[yellow]+{m['swap_grew_mb']} MB during the run[/yellow]")
    console.print(t)
    for w in th["windows"][:5]:
        console.print(f"[yellow]Window {w['window']}: {'; '.join(w['reasons'])}[/yellow]")
    if s.get("warmup_windows_excluded"):
        console.print("[dim]The first window is treated as warm-up and left out of the "
                      "trends.[/dim]")
    if not s.get("all_samples_kept", True):
        console.print("[yellow]More than 2M iterations: final statistics use the first 2M; "
                      "every window's own numbers are in the .jsonl.[/yellow]")
    _print_integrity_issues([])
