"""
edgelens.cli
-------------
Command-line interface: doctor, monitor, benchmark, diagnose, report, compare.
"""

import json
import sys
import time
from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .benchmark.fingerprint import build_fingerprint
from .benchmark.onnx_pipeline import parse_input_shapes
from .benchmark.runner import run_benchmark
from .compare.engine import compare as run_compare
from .diagnose.engine import diagnose as run_diagnose
from .validate.engine import exit_code as validate_exit_code
from .validate.engine import validate as run_validate
from .hardware import detector, power, readiness, telemetry
from .report.generator import generate_html_report

def _never_crash_on_console_encoding():
    """Output uses →, ✓, ⚠ and em dashes. Under a non-UTF-8 locale with output
    piped (CI logs, `edgelens ... > steps.log`), printing one of them raised
    UnicodeEncodeError and lost the run's console output. Replace instead."""
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if enc and enc != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass


_never_crash_on_console_encoding()

app = typer.Typer(
    help="EdgeLens — measure, diagnose and validate AI pipelines on edge devices "
         "(deep NVIDIA Jetson support).",
    no_args_is_help=True,
)
console = Console()

DEMO_WARNING = (
    "SIMULATED DATA — NOT A REAL MEASUREMENT. Every number below is synthetic, "
    "generated only to preview EdgeLens' output format."
)


def _demo_banner():
    console.print(Panel(f"⚠️  {DEMO_WARNING}", style="bold red", border_style="red"))


def _non_jetson_notice():
    console.print(Panel(
        "Not a Jetson device. Real measurements still work here: "
        "[bold]--model[/bold], [bold]--pipeline[/bold] and observer mode "
        "(edgelens.trace) measure latency, tail, deadlines and CPU/RAM on any Linux "
        "host. Jetson-only telemetry (GPU load, board power/energy, power mode) is "
        "unavailable. A bare `edgelens benchmark` with no model or pipeline runs "
        "[bold]demo mode[/bold] (synthetic data, clearly labelled).",
        title="Notice", style="yellow", border_style="yellow",
    ))


def _tensorrt_cell(sw):
    version = sw.get("tensorrt")
    if not version:
        return "[dim]not detected[/dim]"
    if sw.get("tensorrt_python_bindings", True):
        return version
    return (
        f"{version} [yellow](system library via {sw.get('tensorrt_source')}; "
        f"`import tensorrt` not available in Python {sw['python']} — "
        f"apt bindings target the system Python only)[/yellow]"
    )


def _ort_providers_cell(is_jetson):
    try:
        from .benchmark.onnx_pipeline import available_providers, unloadable_providers
        providers = available_providers()
        unloadable = unloadable_providers()
    except Exception:  # broken/ABI-mismatched onnxruntime must not kill doctor
        providers, unloadable = [], []
    if not providers:
        return "[dim]onnxruntime not installed[/dim]"
    text = ", ".join(p.replace("ExecutionProvider", "") for p in providers)
    if unloadable:
        names = ", ".join(p.replace("ExecutionProvider", "") for p in unloadable)
        text += f" [yellow]({names} listed but cannot load — see Benchmark readiness)[/yellow]"
    elif is_jetson and set(providers) <= {"CPUExecutionProvider", "AzureExecutionProvider"}:
        text += " [yellow](CPU-only build — `--model` benchmarks will not use the GPU)[/yellow]"
    return text


def _exit_on_write_error(path, e):
    """Explain a failed write in one line (no traceback) and exit 1.
    Typical case: the file was created by an earlier `sudo edgelens ...` run
    and is owned by root."""
    hint = ""
    if isinstance(e, PermissionError) and Path(path).exists():
        hint = (f"\n'{path}' already exists and is not writable by you; if an earlier "
                f"run used sudo it is owned by root. Fix:  sudo chown $USER {path}  "
                f"or save elsewhere.")
    console.print(f"[red]Could not write {path}: {e.strerror or e}[/red]{hint}")
    raise typer.Exit(1)


def _write_or_exit(path, text):
    """Write an output file, or explain the failure in one line (no traceback)."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
    except OSError as e:
        _exit_on_write_error(path, e)


def _load_result_or_exit(path):
    """Read an EdgeLens result JSON, or explain in one line why it can't be
    used (missing, unreadable, not JSON, not a result) and exit 1."""
    path = Path(path)
    if not path.is_file():
        console.print(f"[red]No such file: {path}[/red] — run `edgelens benchmark` first.")
        raise typer.Exit(1)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as e:
        console.print(f"[red]Cannot read {path} as an EdgeLens result: "
                      f"{type(e).__name__}: {e}[/red]")
        raise typer.Exit(1)
    if not isinstance(data, dict) or "stage_avg_ms" not in data:
        console.print(f"[red]{path} is JSON but not an EdgeLens benchmark/trace result "
                      f"(no 'stage_avg_ms'). Pass the file written by `edgelens benchmark` "
                      f"or `edgelens.trace(..., save=...)`.[/red]")
        raise typer.Exit(1)
    return data


def _ms(v):
    return "n/a" if v is None else f"{v:.3f}"


def _print_benchmark(result):
    pipe = result.get("pipeline") or {}
    roles = {s["name"]: s.get("role", "") for s in pipe.get("stages", [])}
    stats = result.get("stage_stats_ms") or {}
    table = Table(title=f"EdgeLens Benchmark ({result['mode']} mode, {pipe.get('pack', 'custom')} "
                        f"pack, {result['pipeline_source']})")
    table.add_column("Stage")
    table.add_column("Role", style="dim")
    table.add_column("Mean")
    table.add_column("p99", style="dim")
    table.add_column("Share")
    stage_sum = sum(result["stage_avg_ms"].values()) or 1e-9
    for stage, ms in result["stage_avg_ms"].items():
        pct = (ms / stage_sum) * 100
        bar = "#" * max(1, int(pct / 4))
        table.add_row(stage, roles.get(stage, ""), f"{ms:.3f} ms",
                      _ms((stats.get(stage) or {}).get("p99")), f"{bar} {pct:.1f}%")
    console.print(table)

    for fb in (pipe.get("config") or {}).get("provider_fallbacks") or []:
        console.print(f"Skipped {fb['provider']}: {fb['reason']}", style="yellow",
                      markup=False, highlight=False)

    lat = result.get("latency") or {}
    console.print(f"[bold]End-to-end:[/bold] mean {_ms(lat.get('mean'))} ms   "
                  f"[bold]Throughput:[/bold] {result['fps']} it/s   "
                  f"({result['iterations']} iterations)")
    console.print(f"min {_ms(lat.get('min'))} · p50 {_ms(lat.get('p50'))} · "
                  f"p95 {_ms(lat.get('p95'))} · p99 {_ms(lat.get('p99'))} · "
                  f"p99.9 {_ms(lat.get('p99_9'))} · max {_ms(lat.get('max'))} · "
                  f"jitter {_ms(lat.get('jitter'))} ms")
    if lat.get("insufficient_samples"):
        console.print(f"[dim]Not reported (too few samples): "
                      f"{'; '.join(lat['insufficient_samples'])}[/dim]")

    dl = result.get("deadline")
    if dl:
        style = "green" if dl["met"] else "red"
        status = "MET" if dl["met"] else "MISSED"
        console.print(Panel(
            f"[bold]{status}[/bold]  deadline {dl['deadline_ms']} ms · misses "
            f"{dl['misses']}/{dl['iterations']} ({dl['miss_ratio']*100:.3f}%) · worst "
            f"{dl['worst_ms']} ms (overrun {dl['worst_overrun_ms']} ms) · longest burst "
            f"{dl['max_consecutive_misses']}",
            title="Deadline", border_style=style, style=style))
    bl = result.get("backlog")
    if bl:
        console.print(f"[dim]Backlog at {bl['period_ms']} ms period ({bl['method']}): "
                      f"utilization {bl['utilization']}, stable={bl['stable']}, "
                      f"max backlog {bl['max_backlog_items']} item(s)[/dim]")

    pm = result.get("pack_metrics") or {}
    if pm.get("real_time_factor"):
        r = pm["real_time_factor"]
        console.print(f"[bold]Real-time factor:[/bold] mean {r['mean']} · p99 {r['p99']} · "
                      f"max {r['max']} (service time / {pm['period_ms']} ms period, headroom "
                      f"{pm['headroom_pct']}%)")

    en = result.get("energy") or {}
    pw = (result.get("telemetry") or {}).get("power") or {}
    if not en.get("available") and pw.get("reason") == "permission_denied":
        console.print(f"[yellow]Power: sensor is root-only, not measured. Fix: "
                      f"{pw.get('fix')}  (details: edgelens doctor)[/yellow]")
    if en.get("available"):
        line = (f"[bold]Power:[/bold] {en['power_w_mean']} W mean · "
                f"[bold]Energy:[/bold] {en['energy_per_iteration_j']*1000:.2f} mJ/iteration · "
                f"{en['iterations_per_joule']} iterations/J [dim]({en['method']})[/dim]")
        if en.get("dynamic_energy_per_iteration_j") is not None:
            line += (f" · dynamic {en['dynamic_energy_per_iteration_j']*1000:.2f} mJ/it "
                     f"over {en['idle_power_w']} W idle")
        console.print(line)
        if en.get("low_confidence"):
            console.print(f"[yellow]Energy: {en['low_confidence']}[/yellow]")

    tel = result.get("telemetry", {})
    if tel:
        console.print(
            f"[dim]Telemetry over {tel.get('sample_count', 0)} samples "
            f"({tel.get('duration_s', 0)}s, sampler {tel.get('sampler_cpu_ms_mean')} ms CPU/sample): "
            f"CPU mean {tel.get('cpu_percent_mean')}% / peak {tel.get('cpu_percent_peak')}% · "
            f"GPU mean {tel.get('gpu_percent_mean')}% / peak {tel.get('gpu_percent_peak')}% · "
            f"peak temp {tel.get('max_temp_c')}C[/dim]"
        )
    ident = result.get("identity") or {}
    if ident:
        console.print(f"[dim]environment_id {ident.get('environment_id')} · experiment_id "
                      f"{ident.get('experiment_id')} · run_id {ident.get('run_id')}[/dim]")


@app.command()
def doctor():
    """Environment + hardware fingerprint."""
    fp = detector.full_fingerprint()
    hw, sw = fp["hardware"], fp["software"]

    table = Table(title=f"EdgeLens v{__version__} — Doctor", show_header=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Board", hw["model"])
    table.add_row("Is Jetson", "[green]yes[/green]" if hw["is_jetson"] else "[yellow]no[/yellow]")
    table.add_row("L4T / JetPack", sw["jetpack_l4t"] or "[dim]not detected[/dim]")
    table.add_row("CUDA", sw["cuda"] or "[dim]not detected[/dim]")
    table.add_row("TensorRT", _tensorrt_cell(sw))
    table.add_row("ORT providers", _ort_providers_cell(hw["is_jetson"]))
    table.add_row("Python", sw["python"])
    table.add_row("OS", sw["os"])
    console.print(table)

    pkg_table = Table(title="AI / vision packages")
    pkg_table.add_column("Package")
    pkg_table.add_column("Version")
    for pkg, ver in sw["packages"].items():
        pkg_table.add_row(pkg, ver or "[dim]not installed[/dim]")
    console.print(pkg_table)

    src = _print_telemetry_sources()
    _print_readiness(hw["is_jetson"], src["power_ina3221"])

    if not hw["is_jetson"]:
        _non_jetson_notice()


def _print_readiness(is_jetson, power_probe):
    """Benchmark readiness: is this board in a state that gives trustworthy,
    comparable numbers? (clocks, power mode, GPU providers, power sensor)"""
    checks = readiness.gather(is_jetson, power_probe)
    icon = {"ok": "[green]✓[/green]", "warn": "[yellow]![/yellow]", "info": "[dim]·[/dim]"}
    t = Table(title="Benchmark readiness")
    t.add_column("")
    t.add_column("Check")
    t.add_column("State")
    t.add_column("What it means", style="dim")
    for c in checks:
        t.add_row(icon[c["status"]], c["name"], c["value"], c["advice"] or "")
    console.print(t)
    fixes = [c for c in checks if c["status"] == "warn" and c.get("fix")]
    if fixes:
        console.print("[bold]To fix:[/bold]")
        for c in fixes:
            # soft_wrap + no markup: the command stays on one copyable line
            console.print(f"  # {c['name']}", style="dim")
            console.print(f"  {c['fix']}", soft_wrap=True, markup=False, highlight=False)
    status, text = readiness.summary(checks, is_jetson)
    style = {"ok": "green", "warn": "yellow", "info": "dim"}[status]
    console.print(f"[{style}]{text}[/{style}]")


def _print_telemetry_sources():
    """Which telemetry sources work here — the first place to look when GPU%,
    power or energy show up as n/a in a benchmark."""
    src = telemetry.probe_sources()
    t = Table(title="Telemetry sources")
    t.add_column("Source")
    t.add_column("Status")
    t.add_column("Value")
    t.add_column("Read cost", style="dim")

    g = src["gpu_load_sysfs"]
    t.add_row("GPU load (sysfs)",
              f"[green]ok[/green] [dim]{g['path']}[/dim]" if g["value_percent"] is not None
              else "[yellow]not found[/yellow]",
              f"{g['value_percent']}%" if g["value_percent"] is not None else "-",
              f"{g['read_ms']} ms")

    pw = src["power_ina3221"]
    if pw["total_w"] is not None:
        status, value = "[green]ok[/green]", f"{pw['total_w']:.2f} W ({pw['method']})"
    elif pw.get("permission_denied"):
        status, value = "[yellow]found, root-only[/yellow]", "fix under Benchmark readiness"
    elif pw["rails_found"]:
        status, value = f"[red]{pw['error']}[/red]", ", ".join(pw["rails_found"])
    else:
        status, value = "[yellow]no INA3221 in sysfs[/yellow]", "-"
    t.add_row("Power (INA3221 sysfs)", status, value, f"{pw['read_ms']} ms")

    th = src["thermal_zones"]
    t.add_row("Thermal zones", "[green]ok[/green]" if th["count"] else "[yellow]none[/yellow]",
              f"{th['count']} zones, max {th['max_c']} C" if th["count"] else "-",
              f"{th['read_ms']} ms")

    tg = src["tegrastats"]
    if not tg["available"]:
        t.add_row("tegrastats", "[dim]not installed[/dim]", "-", "-")
    elif tg["first_line_s"] is None:
        t.add_row("tegrastats", "[red]no output within 3 s[/red]", "-", "-")
    else:
        rails = ", ".join(f"{k} {v:.0f} mW" for k, v in tg["rails_mw"].items()) or "no power rails"
        t.add_row("tegrastats", "[green]ok[/green]", f"GPU {tg['gpu_percent']}% · {rails}",
                  f"first line {tg['first_line_s']} s")
    console.print(t)
    return src


@app.command()
def monitor(
    duration: int = typer.Option(10, help="Seconds to monitor."),
    interval: float = typer.Option(1.0, help="Seconds between samples."),
):
    """Live terminal dashboard (thin wrapper over psutil / tegrastats)."""
    if not detector.is_jetson():
        console.print("[yellow]Non-Jetson host — showing CPU/RAM/temp only "
                       "(no GPU/tegrastats data available here).[/yellow]")

    start = time.time()
    reader = power.PowerReader()
    stream = None
    if telemetry.read_gpu_percent_sysfs() is None and telemetry.TegrastatsStream.available():
        stream = telemetry.TegrastatsStream(interval_ms=int(interval * 1000))
        stream.start()
    with Live(console=console, refresh_per_second=4) as live:
        while time.time() - start < duration:
            snap = telemetry.snapshot(stream=stream, power_reader=reader if reader.available() else None)
            table = Table(title="EdgeLens Monitor")
            table.add_column("Metric")
            table.add_column("Value")
            table.add_row("CPU", f"{snap['cpu_percent']:.1f}%")
            table.add_row("GPU", f"{snap['gpu_percent']:.1f}%" if snap["gpu_percent"] is not None else "[dim]n/a[/dim]")
            table.add_row("RAM", f"{snap['mem_used_gb']}/{snap['mem_total_gb']} GB ({snap['mem_percent']:.1f}%)")
            if snap.get("power_w") is not None:
                rails = ", ".join(f"{k} {v:.2f}W" for k, v in snap["rails_w"].items())
                table.add_row("Power", f"{snap['power_w']:.2f} W  [dim]({rails})[/dim]")
            for name, val in snap["temps_c"].items():
                table.add_row(f"Temp [{name}]", f"{val:.1f}C")
            live.update(table)
            time.sleep(interval)
    if stream is not None:
        stream.stop()


@app.command()
def benchmark(
    iterations: int = typer.Option(50, min=1, help="Number of timed iterations."),
    model: str = typer.Option(None, "--model", help="Path to a real .onnx model — runs a "
                               "real ONNX Runtime session (CPU on a laptop, CUDA/TensorRT "
                               "on Jetson if available)."),
    pipeline: str = typer.Option(None, "--pipeline", help="Path to a Python script "
                                  "defining build_pipeline() (returns an "
                                  "edgelens.Pipeline) or build_stage_fns() (returns a "
                                  "dict of stage functions) — for a real camera, a "
                                  "non-ONNX runtime, or any custom stages. See "
                                  "examples/timeseries_vibration.py."),
    provider: str = typer.Option(None, "--provider", help="Override the ONNX Runtime "
                                  "execution provider, e.g. CUDAExecutionProvider. "
                                  "Only applies with --model."),
    input_shape: List[str] = typer.Option(None, "--input-shape", help="Concrete shape "
                                           "for a dynamic model input, e.g. 1x8x2048 or "
                                           "vib:1x8x2048. Repeat for multi-input models. "
                                           "Only applies with --model."),
    demo: bool = typer.Option(False, "--demo", help="Force synthetic/demo data."),
    scenario: str = typer.Option(
        "balanced", help="Demo scenario: balanced|preprocess|memory|gpu|thermal"
    ),
    save: str = typer.Option("edgelens_benchmark.json", help="Output JSON path."),
    deadline_ms: float = typer.Option(None, "--deadline-ms", help="Per-iteration deadline: "
                                      "report misses, miss ratio, worst overrun and miss "
                                      "bursts."),
    period_ms: float = typer.Option(None, "--period-ms", help="Input period (e.g. 33.3 for "
                                    "a 30 FPS camera). Without --pace, estimates the "
                                    "backlog at that rate."),
    pace: bool = typer.Option(False, "--pace", help="Release iterations every --period-ms "
                              "and measure true response time, including queueing."),
    idle_baseline: float = typer.Option(0.0, "--idle-baseline", help="Seconds of idle power "
                                        "measurement before the run, to report dynamic "
                                        "(workload-only) energy. Needs INA3221 (Jetson)."),
):
    """Run a benchmark; measure per-stage latency, tail, deadlines, power and energy."""
    is_jetson = detector.is_jetson()

    if not demo and model is None and pipeline is None and not is_jetson:
        console.print("[yellow]No --model or --pipeline given on a non-Jetson host — "
                       "running in --demo mode automatically. Pass --model path/to/"
                       "model.onnx or --pipeline path/to/script.py for a real "
                       "measurement.[/yellow]")

    # Create the output folder BEFORE the run: a long benchmark must never
    # finish and then lose its result to a missing directory.
    try:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        console.print(f"[red]Cannot create output folder for {save}: {e}[/red]")
        raise typer.Exit(1)

    with console.status("Running benchmark..."):
        try:
            result = run_benchmark(
                iterations=iterations, demo=demo, demo_scenario=scenario,
                model_path=model, provider=provider, pipeline_path=pipeline,
                input_shapes=parse_input_shapes(input_shape),
                deadline_ms=deadline_ms, period_ms=period_ms, pace=pace,
                idle_baseline_s=idle_baseline or None,
            )
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    if result["mode"] == "demo":
        _demo_banner()

    _print_benchmark(result)

    _write_or_exit(Path(save), json.dumps(result, indent=2))
    console.print(f"[green]Saved →[/green] {save}")

    if result["mode"] == "demo":
        _demo_banner()


@app.command()
def diagnose(
    benchmark_file: str = typer.Argument("edgelens_benchmark.json", help="Path to a benchmark JSON file."),
):
    """Evidence-based verdict: deadline, bottleneck stage/resource, thermal, memory,
    clocks, plus scenario-pack findings."""
    path = Path(benchmark_file)
    data = _load_result_or_exit(path)
    if data.get("mode") == "demo":
        _demo_banner()

    verdict = run_diagnose(data)
    primary = verdict["primary"]

    evidence_lines = "\n".join(
        f"  {k}: {v}" for k, v in (primary.get("evidence") or {}).items()
    )
    console.print(Panel(
        f"[bold]{primary['type']}[/bold]  [dim]({primary['evidence_strength']} evidence)[/dim]\n\n"
        f"{primary['detail']}\n\n"
        f"[dim]Evidence:\n{evidence_lines}[/dim]\n\n"
        f"[cyan]→ {primary['recommendation']}[/cyan]",
        title="Diagnosis",
    ))
    for f in verdict["secondary"]:
        console.print(f"  also considered: [bold]{f['type']}[/bold] "
                       f"({f['evidence_strength']} evidence) — {f['detail']}")

    out = path.with_suffix("").with_suffix(".diagnosis.json")
    _write_or_exit(out, json.dumps(verdict, indent=2))
    console.print(f"[green]Saved →[/green] {out}")

    if data.get("mode") == "demo":
        _demo_banner()


@app.command()
def report(
    benchmark_file: str = typer.Argument("edgelens_benchmark.json", help="Path to a benchmark JSON file."),
    output: str = typer.Option("edgelens_report.html", help="Output HTML path."),
):
    """Generate a self-contained HTML report (fingerprint + benchmark + diagnosis)."""
    path = Path(benchmark_file)
    bench = _load_result_or_exit(path)
    hw_sw_fp = detector.full_fingerprint()
    verdict = run_diagnose(bench)
    fingerprint = build_fingerprint(hw_sw_fp, bench, verdict)

    try:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        out_path = generate_html_report(fingerprint, bench, verdict, output)
    except OSError as e:
        _exit_on_write_error(output, e)
    fp_json_path = Path(output).with_suffix(".fingerprint.json")
    _write_or_exit(fp_json_path, json.dumps(fingerprint, indent=2))

    console.print(f"[green]Report saved →[/green] {out_path}")
    console.print(f"[green]Fingerprint saved →[/green] {fp_json_path}  "
                   f"[dim](environment_id: {fingerprint['fingerprint_id']})[/dim]")

    if bench.get("mode") == "demo":
        _demo_banner()
        console.print("[dim]The HTML report itself is also watermarked as simulated.[/dim]")


@app.command()
def compare(
    before_file: str = typer.Argument(..., help="Path to the 'before' benchmark JSON file."),
    after_file: str = typer.Argument(..., help="Path to the 'after' benchmark JSON file."),
    strict_env: bool = typer.Option(False, "--strict-env", help="Exit with code 2 when the "
                                    "two runs come from different environments (power "
                                    "mode, clocks, board, JetPack, versions)."),
):
    """Compare two benchmark runs (before/after) and flag regressions."""
    before = _load_result_or_exit(before_file)
    after = _load_result_or_exit(after_file)

    if before.get("mode") == "demo" or after.get("mode") == "demo":
        console.print("[yellow]Note: comparing one or more --demo (simulated) results — "
                       "this comparison is not meaningful for real hardware decisions.[/yellow]")

    result = run_compare(before, after)

    env = result["environment"]
    if env["comparable"] is False:
        lines = "\n".join(f"  {d['label']}: {d['before']} -> {d['after']}"
                          for d in env["differences"])
        console.print(Panel(
            "These runs were taken in DIFFERENT environments, so differences below may "
            "come from the environment, not from your change:\n" + lines,
            title="⚠️  Environment mismatch", style="yellow", border_style="yellow"))
    if result.get("requirements_differences"):
        lines = "\n".join(f"  {d['label']}: {d['before']} -> {d['after']}"
                          for d in result["requirements_differences"])
        console.print(Panel(
            "These runs measured DIFFERENT requirements, so deadline/miss figures are "
            "not comparable:\n" + lines,
            title="ℹ️  Different experiments", style="cyan", border_style="cyan"))
    if env["comparable"] is None:
        console.print("[dim]Environment not recorded in one or both files (pre-v0.1.0); "
                      "cannot check that the runs are comparable.[/dim]")

    table = Table(title="EdgeLens Compare")
    table.add_column("Metric")
    table.add_column("Before")
    table.add_column("After")
    table.add_column("Change")
    for name, d in result["metrics"].items():
        change = f"{d['pct_change']:+.1f}%" if d["pct_change"] is not None else "n/a"
        table.add_row(name, str(d["before"]), str(d["after"]), change)
    console.print(table)

    stage_table = Table(title="Per-stage change")
    stage_table.add_column("Stage")
    stage_table.add_column("Before (ms)")
    stage_table.add_column("After (ms)")
    stage_table.add_column("Change")
    for stage, d in result["stages"].items():
        if d["status"] != "both":
            stage_table.add_row(stage, _ms(d["before_ms"]), _ms(d["after_ms"]), d["status"])
            continue
        change = f"{d['pct_change']:+.1f}%" if d["pct_change"] is not None else "n/a"
        stage_table.add_row(stage, f"{d['before_ms']:.3f}", f"{d['after_ms']:.3f}", change)
    console.print(stage_table)

    if result["verdict"] == "REGRESSION":
        console.print(Panel(
            "\n".join(result["reasons"]),
            title="❌ REGRESSION", style="bold red", border_style="red",
        ))
    else:
        console.print(Panel("No regression detected.", title="✅ PASS", style="bold green",
                             border_style="green"))
    # Exit codes: 2 = not comparable (with --strict-env), 1 = regression, 0 = pass.
    if strict_env and env["comparable"] is False:
        raise typer.Exit(2)
    if result["verdict"] == "REGRESSION":
        raise typer.Exit(1)


@app.command()
def validate(
    benchmark_file: str = typer.Argument("edgelens_benchmark.json",
                                         help="Path to a benchmark/trace result JSON file."),
    p50_ms: float = typer.Option(None, "--p50-ms", help="Max end-to-end p50 latency (ms)."),
    p95_ms: float = typer.Option(None, "--p95-ms", help="Max end-to-end p95 latency (ms)."),
    p99_ms: float = typer.Option(None, "--p99-ms", help="Max end-to-end p99 latency (ms). "
                                 "Needs >= 100 iterations."),
    p99_9_ms: float = typer.Option(None, "--p99.9-ms", help="Max end-to-end p99.9 latency "
                                   "(ms). Needs >= 1000 iterations."),
    max_latency_ms: float = typer.Option(None, "--max-latency-ms",
                                         help="Max worst-case (observed) latency (ms)."),
    max_miss_ratio: float = typer.Option(None, "--max-miss-ratio", help="Max deadline miss "
                                         "ratio, e.g. 0.001 for 0.1%. The run must have a "
                                         "deadline (benchmark --deadline-ms)."),
    min_fps: float = typer.Option(None, "--min-fps", help="Min throughput (iterations/s)."),
    max_power_w: float = typer.Option(None, "--max-power-w", help="Max mean board power (W)."),
    max_energy_mj: float = typer.Option(None, "--max-energy-mj",
                                        help="Max energy per iteration (mJ)."),
):
    """Check a run against requirements: PASS / FAIL / INCONCLUSIVE, with evidence.

    Exit codes: 0 PASS, 1 FAIL, 2 INCONCLUSIVE (a requirement this run cannot
    measure, or a --demo result)."""
    path = Path(benchmark_file)
    data = _load_result_or_exit(path)
    reqs = {"p50_ms": p50_ms, "p95_ms": p95_ms, "p99_ms": p99_ms, "p99_9_ms": p99_9_ms,
            "max_latency_ms": max_latency_ms, "max_miss_ratio": max_miss_ratio,
            "min_fps": min_fps, "max_power_w": max_power_w, "max_energy_mj": max_energy_mj}
    try:
        verdict = run_validate(data, reqs)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if data.get("mode") == "demo":
        _demo_banner()

    def fmt(v, unit):
        if v is None:
            return "—"
        if unit == "":
            return f"{v * 100:.3f}%"
        return f"{v:g} {unit}"

    style = {"PASS": "green", "FAIL": "red", "NOT_MEASURED": "yellow"}
    table = Table(title="Requirements")
    table.add_column("Requirement")
    table.add_column("Limit")
    table.add_column("Measured")
    table.add_column("Result")
    for r in verdict["checks"]:
        op = "≤" if r["direction"] == "max" else "≥"
        status = f"[{style[r['status']]}]{r['status'].replace('_', ' ')}[/{style[r['status']]}]"
        if r.get("reason"):
            status += f"\n[dim]{r['reason']}[/dim]"
        elif r.get("detail"):
            status += f"\n[dim]{r['detail']}[/dim]"
        table.add_row(r["label"], f"{op} {fmt(r['limit'], r['unit'])}",
                      fmt(r["measured"], r["unit"]), status)
    console.print(table)

    if verdict["verdict"] == "FAIL":
        ls = verdict.get("limiting_stage")
        d = verdict["diagnosis"]["primary"]
        body = ""
        if ls:
            tail = (f"p99 {ls['p99_ms']} ms" if ls["basis"] == "p99"
                    else f"mean {ls['mean_ms']} ms")
            body += f"[bold]Limiting stage:[/bold] {ls['stage']} ({tail})\n\n"
        evidence = "\n".join(f"  {k}: {v}" for k, v in (d.get("evidence") or {}).items())
        body += (f"[bold]Why:[/bold] {d['type']} ({d['evidence_strength']} evidence)\n"
                 f"{d['detail']}\n[dim]{evidence}[/dim]\n\n"
                 f"[cyan]Next experiment → {d['recommendation']}[/cyan]")
        console.print(Panel(body, title="❌ REQUIREMENTS NOT MET", border_style="red"))
    elif verdict["verdict"] == "PASS":
        console.print(Panel("Every requirement was measured and met.",
                            title="✅ PASS", style="bold green", border_style="green"))
    else:
        console.print(Panel("\n".join(verdict["notes"]),
                            title="⚠️  INCONCLUSIVE", style="yellow", border_style="yellow"))

    run = verdict["run"]
    console.print(f"[dim]{run['iterations']} iterations · {run.get('board') or 'unknown board'}"
                  f"{' · ' + run['power_mode'] if run.get('power_mode') else ''} · "
                  f"environment_id {run.get('environment_id')} · run_id {run.get('run_id')}[/dim]")

    out = path.with_suffix("").with_suffix(".validation.json")
    _write_or_exit(out, json.dumps(verdict, indent=2))
    console.print(f"[green]Saved →[/green] {out}")
    raise typer.Exit(validate_exit_code(verdict["verdict"]))


@app.command()
def version():
    """Show the EdgeLens version."""
    console.print(f"EdgeLens v{__version__}")


if __name__ == "__main__":
    app()