"""
edgelens.cli
-------------
Command-line interface: doctor, monitor, benchmark, diagnose, report.
"""

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .benchmark.fingerprint import build_fingerprint
from .benchmark.runner import run_benchmark
from .diagnose.engine import diagnose as run_diagnose
from .hardware import detector, telemetry
from .report.generator import generate_html_report

app = typer.Typer(
    help="EdgeLens — an open-source performance doctor for NVIDIA Jetson AI workloads.",
    no_args_is_help=True,
)
console = Console()


def _non_jetson_notice():
    console.print(Panel(
        "This doesn't look like a Jetson device. EdgeLens v0.1 targets NVIDIA "
        "Jetson boards specifically. Commands below will run in [bold]demo mode[/bold] "
        "(synthetic data) so you can preview the tool and its output format — "
        "see README.md for real-hardware usage.",
        title="Notice", style="yellow", border_style="yellow",
    ))


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
    table.add_row("TensorRT", sw["tensorrt"] or "[dim]not detected[/dim]")
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
    demo: bool = typer.Option(False, "--demo", help="Force synthetic/demo data."),
    scenario: str = typer.Option(
        "balanced", help="Demo scenario: balanced|preprocess|memory|gpu|thermal"
    ),
    save: str = typer.Option("edgelens_benchmark.json", help="Output JSON path."),
):
    """Run a benchmark; measure per-stage latency, FPS, and utilization."""
    is_jetson = detector.is_jetson()
    use_demo = demo or not is_jetson
    if use_demo and not demo:
        console.print("[yellow]Non-Jetson host detected — running in --demo mode automatically.[/yellow]")

    with console.status("Running benchmark..."):
        result = run_benchmark(iterations=iterations, demo=use_demo, demo_scenario=scenario)

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

    Path(save).write_text(json.dumps(result, indent=2))
    console.print(f"[green]Saved →[/green] {save}")

    if result["mode"] == "demo":
        console.print("[dim]Note: this is synthetic demo data, not a real hardware measurement.[/dim]")


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
    verdict = run_diagnose(data)
    primary = verdict["primary"]

    console.print(Panel(
        f"[bold]{primary['type']}[/bold]  [dim](confidence {primary['confidence']*100:.0f}%)[/dim]\n\n"
        f"{primary['detail']}\n\n"
        f"[cyan]→ {primary['recommendation']}[/cyan]",
        title="Diagnosis",
    ))
    for f in verdict["secondary"]:
        console.print(f"  also considered: [bold]{f['type']}[/bold] "
                       f"({f['confidence']*100:.0f}%) — {f['detail']}")

    out = path.with_suffix("").with_suffix(".diagnosis.json")
    out.write_text(json.dumps(verdict, indent=2))
    console.print(f"[green]Saved →[/green] {out}")


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


@app.command()
def version():
    """Show the EdgeLens version."""
    console.print(f"EdgeLens v{__version__}")


if __name__ == "__main__":
    app()
