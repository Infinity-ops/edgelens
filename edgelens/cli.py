"""
edgelens.cli
-------------
Command-line interface: doctor, monitor, benchmark, diagnose, report, compare.
"""

import json
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
from .hardware import detector, power, telemetry
from .report.generator import generate_html_report

app = typer.Typer(
    help="EdgeLens — an open-source performance doctor for NVIDIA Jetson AI workloads.",
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
        from .benchmark.onnx_pipeline import available_providers
        providers = available_providers()
    except Exception:  # broken/ABI-mismatched onnxruntime must not kill doctor
        providers = []
    if not providers:
        return "[dim]onnxruntime not installed[/dim]"
    text = ", ".join(p.replace("ExecutionProvider", "") for p in providers)
    if is_jetson and set(providers) <= {"CPUExecutionProvider", "AzureExecutionProvider"}:
        text += " [yellow](CPU-only build — `--model` benchmarks will not use the GPU)[/yellow]"
    return text


def _write_or_exit(path, text):
    """Write an output file, or explain the failure in one line (no traceback).
    Typical case: the file was created by an earlier `sudo edgelens ...` run
    and is owned by root."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    except OSError as e:
        hint = ""
        if isinstance(e, PermissionError) and Path(path).exists():
            hint = (f"\n'{path}' already exists and is not writable by you; if an earlier "
                    f"run used sudo it is owned by root. Fix:  sudo chown $USER {path}  "
                    f"or save elsewhere with --save.")
        console.print(f"[red]Could not write {path}: {e.strerror or e}[/red]{hint}")
        raise typer.Exit(1)


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

    _print_telemetry_sources()

    if not hw["is_jetson"]:
        _non_jetson_notice()


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
        status, value = "[red]found, root-only[/red]", "see fix below"
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
    if pw.get("permission_denied") and pw.get("fix"):
        console.print(Panel(
            "The board's power sensor (INA3221) exists but is readable by root only, so "
            "power and energy are not reported. Make it readable (until the next reboot):\n\n"
            f"  {pw['fix']}\n\n"
            "Or run a single benchmark as root:  sudo $(which edgelens) benchmark ... "
            "(files it saves will then be owned by root).",
            title="Power: permission needed", style="yellow", border_style="yellow"))


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
    iterations: int = typer.Option(50, help="Number of timed iterations."),
    model: str = typer.Option(None, "--model", help="Path to a real .onnx model — runs a "
                               "real ONNX Runtime session (CPU on a laptop, CUDA/TensorRT "
                               "on Jetson if available)."),
    pipeline: str = typer.Option(None, "--pipeline", help="Path to a Python script "
                                  "defining build_stage_fns() — for a real camera, a "
                                  "non-ONNX runtime, or anything --model's plain ONNX "
                                  "forward pass can't express. See "
                                  "tests/fixtures/example_pipeline.py."),
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
    """Rule-based bottleneck verdict: CPU-bound / GPU-bound / thermal / memory-bound."""
    path = Path(benchmark_file)
    if not path.exists():
        console.print(f"[red]No such file: {benchmark_file}[/red] — run `edgelens benchmark` first.")
        raise typer.Exit(1)

    data = json.loads(path.read_text())
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
    if not path.exists():
        console.print(f"[red]No such file: {benchmark_file}[/red] — run `edgelens benchmark` first.")
        raise typer.Exit(1)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    hw_sw_fp = detector.full_fingerprint()
    bench = json.loads(path.read_text())
    verdict = run_diagnose(bench)
    fingerprint = build_fingerprint(hw_sw_fp, bench, verdict)

    out_path = generate_html_report(fingerprint, bench, verdict, output)
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
    before_path, after_path = Path(before_file), Path(after_file)
    for p in (before_path, after_path):
        if not p.exists():
            console.print(f"[red]No such file: {p}[/red]")
            raise typer.Exit(1)

    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())

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
def version():
    """Show the EdgeLens version."""
    console.print(f"EdgeLens v{__version__}")


if __name__ == "__main__":
    app()