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
from .hardware import detector, telemetry
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
        "This doesn't look like a Jetson device. EdgeLens v0.1 targets NVIDIA "
        "Jetson boards specifically. Commands below will run in [bold]demo mode[/bold] "
        "(synthetic data) so you can preview the tool and its output format — "
        "see README.md for real-hardware usage.",
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

    if not hw["is_jetson"]:
        _non_jetson_notice()


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
    with Live(console=console, refresh_per_second=4) as live:
        while time.time() - start < duration:
            snap = telemetry.snapshot()
            table = Table(title="EdgeLens Monitor")
            table.add_column("Metric")
            table.add_column("Value")
            table.add_row("CPU", f"{snap['cpu_percent']:.1f}%")
            table.add_row("GPU", f"{snap['gpu_percent']:.1f}%" if snap["gpu_percent"] is not None else "[dim]n/a[/dim]")
            table.add_row("RAM", f"{snap['mem_used_gb']}/{snap['mem_total_gb']} GB ({snap['mem_percent']:.1f}%)")
            for name, val in snap["temps_c"].items():
                table.add_row(f"Temp [{name}]", f"{val:.1f}C")
            live.update(table)
            time.sleep(interval)


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
):
    """Run a benchmark; measure per-stage latency, FPS, and utilization."""
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
            )
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    if result["mode"] == "demo":
        _demo_banner()

    table = Table(title=f"EdgeLens Benchmark ({result['mode']} mode, {result['pipeline_source']})")
    table.add_column("Stage")
    table.add_column("Avg latency")
    table.add_column("Share")
    total = result["total_latency_ms"] or 1e-9
    for stage, ms in result["stage_avg_ms"].items():
        pct = (ms / total) * 100
        bar = "#" * max(1, int(pct / 4))
        table.add_row(stage.replace("_", " "), f"{ms:.2f} ms", f"{bar} {pct:.1f}%")
    console.print(table)
    console.print(f"[bold]Total:[/bold] {total:.2f} ms   [bold]FPS:[/bold] {result['fps']}")
    console.print(f"P50 {result['latency_p50_ms']}ms  P95 {result['latency_p95_ms']}ms  "
                   f"P99 {result['latency_p99_ms']}ms")

    tel = result.get("telemetry", {})
    if tel:
        console.print(
            f"[dim]Telemetry over {tel.get('sample_count', 0)} samples "
            f"({tel.get('duration_s', 0)}s): "
            f"CPU mean {tel.get('cpu_percent_mean')}% / peak {tel.get('cpu_percent_peak')}% · "
            f"GPU mean {tel.get('gpu_percent_mean')}% / peak {tel.get('gpu_percent_peak')}% · "
            f"peak temp {tel.get('max_temp_c')}C[/dim]"
        )

    Path(save).write_text(json.dumps(result, indent=2))
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
        f"[bold]{primary['type']}[/bold]  [dim](evidence strength {primary['evidence_strength']*100:.0f}%)[/dim]\n\n"
        f"{primary['detail']}\n\n"
        f"[dim]Evidence:\n{evidence_lines}[/dim]\n\n"
        f"[cyan]→ {primary['recommendation']}[/cyan]",
        title="Diagnosis",
    ))
    for f in verdict["secondary"]:
        console.print(f"  also considered: [bold]{f['type']}[/bold] "
                       f"(evidence strength {f['evidence_strength']*100:.0f}%) — {f['detail']}")

    out = path.with_suffix("").with_suffix(".diagnosis.json")
    out.write_text(json.dumps(verdict, indent=2))
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

    out_path = generate_html_report(hw_sw_fp, bench, verdict, output)
    fp_json_path = Path(output).with_suffix(".fingerprint.json")
    fp_json_path.write_text(json.dumps(fingerprint, indent=2))

    console.print(f"[green]Report saved →[/green] {out_path}")
    console.print(f"[green]Fingerprint saved →[/green] {fp_json_path}  "
                   f"[dim](id: {fingerprint['fingerprint_id']})[/dim]")

    if bench.get("mode") == "demo":
        _demo_banner()
        console.print("[dim]The HTML report itself is also watermarked as simulated.[/dim]")


@app.command()
def compare(
    before_file: str = typer.Argument(..., help="Path to the 'before' benchmark JSON file."),
    after_file: str = typer.Argument(..., help="Path to the 'after' benchmark JSON file."),
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
        if d["before_ms"] is None:
            continue
        change = f"{d['pct_change']:+.1f}%" if d["pct_change"] is not None else "n/a"
        stage_table.add_row(stage, f"{d['before_ms']:.2f}", f"{d['after_ms']:.2f}", change)
    console.print(stage_table)

    if result["verdict"] == "REGRESSION":
        console.print(Panel(
            "\n".join(result["reasons"]),
            title="❌ REGRESSION", style="bold red", border_style="red",
        ))
        raise typer.Exit(1)
    else:
        console.print(Panel("No regression detected.", title="✅ PASS", style="bold green",
                             border_style="green"))


@app.command()
def version():
    """Show the EdgeLens version."""
    console.print(f"EdgeLens v{__version__}")


if __name__ == "__main__":
    app()