"""
edgelens.cli_observe
----------------------
v0.2 commands for systems EdgeLens doesn't run itself:

    edgelens record  [-- CMD ...]     markers from C/C++/Python (+ telemetry)
    edgelens watch   --pid/--name/-- CMD   attach mode: per-process resources
    edgelens import-log LOG --regex   timing from an application's own log
    edgelens include                  path for -I (edgelens_mark.h)

Registered on the main app by edgelens.cli (imported at the end of it).
"""

import json
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from .attach import TargetError, observe, to_result
from .cli import _exit_on_write_error, _print_benchmark, _write_or_exit, app, console
from .markers import INCLUDE_DIR, protocol
from .markers.collector import CollectorError
from .markers.logimport import LogImportError, parse_lines
from .markers.session import spans_to_result

_EXTRA = {"allow_extra_args": True, "ignore_unknown_options": True}


def print_integrity(result):
    """Measurement-integrity notes: lost events, unsynced GPU spans, no latency."""
    m = result.get("markers") or {}
    if m.get("source") == "log":
        console.print(f"[dim]Timing parsed from a log ({m.get('received')} lines, "
                      f"timestamps: {m.get('timestamps')}). {m.get('note')}[/dim]")
        return
    if m:
        received, lost = m.get("received") or 0, m.get("lost") or 0
        senders = m.get("senders") or []
        line = (f"[bold]Markers:[/bold] {received} spans received from "
                f"{len(senders)} sender(s) on {m.get('socket')} · lost {lost}")
        if m.get("malformed"):
            line += f" · malformed {m['malformed']}"
        console.print(line)
        if lost:
            console.print(Panel(
                f"{lost} span(s) were LOST ({(m.get('lost_ratio') or 0)*100:.3f}%): the "
                f"application produced them but the collector never received them "
                f"(receive buffer full, or the collector stopped first). Statistics "
                f"below exclude them. Reduce span rate, or check `edgelens selftest`.",
                title="⚠️  Lost marker events", border_style="yellow", style="yellow"))
        late = [s for s in senders if s.get("seq_before_first")]
        if late:
            console.print(f"[dim]{sum(s['seq_before_first'] for s in late)} span(s) were "
                          f"produced before the collector started (not counted as lost).[/dim]")
        if senders and not all(s.get("final_seen") for s in senders):
            console.print("[dim]Some senders did not send a final message (killed, or "
                          "still running): losses after their last delivered span "
                          "can't be seen.[/dim]")
        g = m.get("gpu_timing") or {}
        gpu_now = (result.get("telemetry") or {}).get("gpu_percent_mean")
        gpu_busy = gpu_now is None or gpu_now > 5
        if g.get("unsynced_inference_spans") and not gpu_busy:
            console.print(f"[dim]{g['unsynced_inference_spans']} inference span(s) ended "
                          f"without GPU sync; fine for CPU inference (GPU was idle). For "
                          f"GPU inference use EL_SPAN_GPU / mark.span(..., sync=...).[/dim]")
        elif g.get("unsynced_inference_spans"):
            console.print(Panel(
                ("GPU load is unknown on this host. " if gpu_now is None else "") +
                f"{g['unsynced_inference_spans']} inference span(s) "
                f"({', '.join(g['unsynced_inference_stages'])}) ended WITHOUT GPU "
                f"synchronisation. If they launch CUDA/TensorRT work, the span measures "
                f"the enqueue, not the inference. Use EL_SPAN_GPU / EL_SPAN_CUDA (C++), "
                f"el_end_synced (C) or mark.span(..., sync=...) (Python). Ignore this "
                f"for CPU-only inference.",
                title="GPU timing not synchronised", border_style="cyan", style="cyan"))
        if not received:
            console.print(f"[yellow]{m.get('note') or 'No marker spans received.'}[/yellow]")


def print_process(result):
    p = result.get("process")
    if not p:
        return
    if not p.get("samples"):
        console.print(f"[yellow]Process {p.get('pid')} ({p.get('name')}): "
                      f"{p.get('note')}[/yellow]")
        return
    t = Table(title=f"Process {p['pid']} ({p['name']})"
                    f"{' + children' if p.get('include_children') else ''}")
    t.add_column("Metric")
    t.add_column("Value")
    t.add_row("CPU (% of one core)", f"mean {p['cpu_percent_mean']} · peak {p['cpu_percent_peak']}")
    at_exit = (f" · {p['rss_mb_at_exit']} MB while shutting down"
               if p.get("rss_mb_at_exit") is not None else "")
    t.add_row("Resident memory", f"{p['rss_mb_start']} → {p['rss_mb_end']} MB "
                                  f"(peak {p['rss_mb_peak']}){at_exit}")
    growth = p.get("rss_growth_mb_per_min")
    t.add_row("Memory growth", "n/a (too few samples)" if growth is None
              else f"{growth:+.2f} MB/min")
    t.add_row("Threads", f"{p['threads_min']}–{p['threads_max']}")
    t.add_row("Context switches /s", f"voluntary {p['ctx_voluntary_per_s_mean']} · "
                                      f"involuntary (preempted) {p['ctx_involuntary_per_s_mean']}"
                                      f" (peak {p['ctx_involuntary_per_s_peak']})")
    t.add_row("Observed for", f"{p['duration_s']} s, {p['samples']} samples, sampler "
                               f"{p['sampler_cpu_ms_mean']} ms CPU/sample")
    console.print(t)
    console.print(f"[dim]GPU: {p['gpu']}.[/dim]")


def _print_board(result):
    tel = result.get("telemetry") or {}
    pw = tel.get("power") or {}
    parts = []
    if tel.get("cpu_percent_mean") is not None:
        parts.append(f"CPU {tel['cpu_percent_mean']}% (peak {tel.get('cpu_percent_peak')})")
    if tel.get("gpu_percent_mean") is not None:
        parts.append(f"GPU {tel['gpu_percent_mean']}% (peak {tel.get('gpu_percent_peak')})")
    if tel.get("max_temp_c"):
        parts.append(f"peak temp {tel['max_temp_c']}C")
    if pw.get("available"):
        parts.append(f"power {pw['power_w_mean']} W mean / {pw['power_w_peak']} W peak "
                     f"({pw.get('method')})")
    if parts:
        console.print("[bold]Board:[/bold] " + " · ".join(parts))
    if not pw.get("available") and pw.get("reason") == "permission_denied":
        console.print(f"[yellow]Power: sensor is root-only. Fix: {pw.get('fix')}[/yellow]")


def _save(result, save):
    _write_or_exit(Path(save), json.dumps(result, indent=2))
    console.print(f"[green]Saved →[/green] {save}")


def _common_observe(cmd, pid, name, duration, socket, markers, children, interval,
                    idle_baseline, process_interval=0.5):
    try:
        return observe(cmd=cmd or None, pid=pid, name=name, duration_s=duration,
                       socket_path=socket, markers=markers, children=children,
                       telemetry_interval_s=interval, idle_baseline_s=idle_baseline or None,
                       process_interval_s=process_interval)
    except (CollectorError, TargetError, RuntimeError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command(context_settings=_EXTRA)
def record(
    ctx: typer.Context,
    socket: str = typer.Option(None, "--socket", help=f"Collector socket (default "
                               f"${protocol.ENV_SOCKET} or {protocol.DEFAULT_SOCKET})."),
    duration: float = typer.Option(None, "--duration", help="Stop after this many seconds "
                                   "(default: when the command exits, or Ctrl-C)."),
    name: str = typer.Option("markers", "--name", help="Pipeline name in the result."),
    pack: str = typer.Option("custom", "--pack", help="Scenario pack (custom|vision|timeseries)."),
    deadline_ms: float = typer.Option(None, "--deadline-ms", help="Per-iteration deadline."),
    period_ms: float = typer.Option(None, "--period-ms", help="Input period (e.g. 1000 for "
                                    "1 Hz inference)."),
    interval: float = typer.Option(0.1, "--interval", help="Telemetry interval (s)."),
    idle_baseline: float = typer.Option(0.0, "--idle-baseline", help="Seconds of idle power "
                                        "before starting, for dynamic energy."),
    save: str = typer.Option("edgelens_record.json", "--save", help="Output JSON path."),
):
    """Collect inference markers from any application (C/C++ edgelens_mark.h,
    Python edgelens.mark) plus board telemetry. Optionally starts the
    application:  edgelens record --deadline-ms 3000 -- ./my_app --config app.yaml"""
    cmd = list(ctx.args)
    try:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _exit_on_write_error(save, e)
    where = protocol.socket_path(socket)
    if cmd and cmd[0] == "sudo" and not any(a == "-E" or a.startswith("--preserve-env")
                                             for a in cmd[1:]):
        console.print("[yellow]sudo resets the environment, so the application won't see "
                      "EDGELENS_SOCKET and its spans won't arrive. Use `sudo -E ...` or "
                      "`sudo --preserve-env=EDGELENS_SOCKET ...`.[/yellow]")
    if cmd:
        console.print(f"[dim]Starting: {' '.join(cmd)} (markers on "
                      f"{where if socket else 'a private socket'})[/dim]")
    else:
        console.print(f"Listening for markers on [bold]{where}[/bold] "
                      f"({'for %ss' % duration if duration else 'Ctrl-C to stop'}). "
                      f"Start your application with EDGELENS_SOCKET={where} if it uses "
                      f"another path.")
    run = _common_observe(cmd, None, None, duration, socket, True, True, interval,
                          idle_baseline)
    result = to_result(run, name=name, pack=pack, deadline_ms=deadline_ms,
                       period_ms=period_ms, mode="markers")
    console.print(f"[dim]Ended: {run.ended_by}"
                  f"{'' if run.exit_code is None else f' (exit code {run.exit_code})'}[/dim]")
    if result.get("iterations"):
        _print_benchmark(result)
    else:
        _print_board(result)
    print_integrity(result)
    print_process(result)
    _print_blocking(result)
    _save(result, save)


def _print_blocking(result):
    from .core import integrity
    from .cli import _print_integrity_issues
    _print_integrity_issues([i for i in integrity.check(result)
                             if i["code"] not in ("MARKER_EVENTS_LOST", "GPU_TIMING_NOT_SYNCED",
                                                  "LATENCY_NOT_MEASURED")])


@app.command(context_settings=_EXTRA)
def watch(
    ctx: typer.Context,
    pid: int = typer.Option(None, "--pid", help="Attach to this process id."),
    name: str = typer.Option(None, "--name", help="Attach to the one process with this "
                             "name (or command-line substring)."),
    duration: float = typer.Option(None, "--duration", help="Stop after this many seconds "
                                   "(default: when the process exits, or Ctrl-C)."),
    children: bool = typer.Option(True, "--children/--no-children", help="Include the "
                                  "target's child processes."),
    markers: bool = typer.Option(True, "--markers/--no-markers", help="Also collect "
                                 "marker spans if the application sends any."),
    socket: str = typer.Option(None, "--socket", help="Marker socket (see `record`)."),
    interval: float = typer.Option(0.1, "--interval", help="Telemetry interval (s)."),
    process_interval: float = typer.Option(0.5, "--process-interval", help="Process "
                                           "sampling interval (s). CPU% resolution is "
                                           "about 1 / (100 x interval) of a core."),
    deadline_ms: float = typer.Option(None, "--deadline-ms", help="Deadline for marker "
                                      "iterations, if any."),
    save: str = typer.Option("edgelens_watch.json", "--save", help="Output JSON path."),
):
    """Attach mode: profile a running application WITHOUT changing its code.
    Per-process CPU, memory, threads, context switches + board power and
    temperature, on one timeline.  edgelens watch --name my_app --duration 60"""
    cmd = list(ctx.args)
    targets = sum(x is not None and x != [] for x in (pid, name, cmd or None))
    if targets != 1:
        console.print("[red]Give exactly one target: --pid N, --name NAME, or -- CMD ...[/red]")
        raise typer.Exit(2)
    try:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _exit_on_write_error(save, e)
    run = _common_observe(cmd, pid, name, duration, socket, markers, children, interval, 0,
                          process_interval)
    result = to_result(run, name=(run.process or {}).get("name") or "watch",
                       deadline_ms=deadline_ms, mode="attach")
    console.print(f"[dim]Ended: {run.ended_by}"
                  f"{'' if run.exit_code is None else f' (exit code {run.exit_code})'}[/dim]")
    print_process(result)
    _print_board(result)
    if result.get("iterations"):
        _print_benchmark(result)
        print_integrity(result)
    else:
        console.print("[dim]Latency: not measured. Attach mode sees resources from "
                      "outside; add markers (edgelens_mark.h) for per-inference timing.[/dim]")
    _print_blocking(result)
    _save(result, save)


@app.command("import-log")
def import_log(
    log_file: str = typer.Argument(..., help="Log file written by the application."),
    regex: str = typer.Option(..., "--regex", help="Regex with a named duration group, "
                              "e.g. 'took (?P<ms>[0-9.]+) ms'. See docs/markers.md."),
    stage: str = typer.Option("inference", "--stage", help="Stage name when the regex has "
                              "no (?P<stage>...) group."),
    name: str = typer.Option("log", "--name", help="Pipeline name in the result."),
    deadline_ms: float = typer.Option(None, "--deadline-ms", help="Per-iteration deadline."),
    save: str = typer.Option("edgelens_log.json", "--save", help="Output JSON path."),
):
    """Build a result from timing lines in an existing log (no code changes)."""
    path = Path(log_file)
    if not path.is_file():
        console.print(f"[red]No such file: {path}[/red]")
        raise typer.Exit(1)
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            spans, info = parse_lines(f, regex, stage=stage)
    except LogImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2)
    if not spans:
        console.print(f"[red]The regex matched no line in {path}.[/red]")
        raise typer.Exit(1)
    result = spans_to_result(spans, info, name=name, deadline_ms=deadline_ms,
                             mode="imported-log", pipeline_source=f"log:{path.name}",
                             source="log")
    _print_benchmark(result)
    print_integrity(result)
    _save(result, save)


@app.command()
def selftest(
    quick: bool = typer.Option(False, "--quick", help="A few seconds instead of ~1 minute "
                               "(less precise)."),
    no_c: bool = typer.Option(False, "--no-c", help="Skip compiling the C header check."),
    save: str = typer.Option("edgelens_selftest.json", "--save", help="Output JSON path."),
):
    """Measure EdgeLens itself on this board: clock, harness accuracy against
    a known reference workload, marker cost and delivery, sampler overhead.
    Run it once per board before trusting the numbers."""
    from . import selftest as st
    with console.status("Measuring EdgeLens on this machine..."):
        doc = st.run(quick=quick, include_c=not no_c)
    table = Table(title=f"EdgeLens selftest — {doc.get('board') or 'this machine'} "
                        f"(Python {doc['python']})")
    for col in ("Check", "Result", "Value", "Limit", "What was measured"):
        table.add_column(col)
    style = {"PASS": "green", "WARN": "yellow", "FAIL": "red", "SKIP": "dim"}
    for c in doc["checks"]:
        val = "-" if c["value"] is None else f"{c['value']} {c['unit']}".strip()
        lim = "-" if c["limit"] is None else str(c["limit"])
        detail = c["detail"] + (f"\n[cyan]Fix: {c['fix']}[/cyan]" if c.get("fix") else "")
        table.add_row(c["check"], f"[{style[c['status']]}]{c['status']}[/{style[c['status']]}]",
                      val, lim, detail)
    console.print(table)
    v = doc["verdict"]
    msg = {"PASS": "EdgeLens's own overhead and accuracy are within limits on this machine.",
           "WARN": "Usable, with the warnings above; numbers near those limits deserve care.",
           "FAIL": "A measurement-integrity check failed: don't trust results from this "
                   "machine until it is fixed."}[v]
    console.print(Panel(msg, title=f"selftest: {v}", border_style=style[v]))
    _save(doc, save)
    if v == "FAIL":
        raise typer.Exit(1)


ros2_app = typer.Typer(help="ROS 2 measurements (experimental in v0.2; needs a sourced "
                             "ROS 2 installation for rclpy).", no_args_is_help=True)
app.add_typer(ros2_app, name="ros2")


@ros2_app.command("latency")
def ros2_latency(
    from_topic: str = typer.Option(..., "--from", help="Input topic, e.g. /sensor/imu."),
    to_topic: str = typer.Option(None, "--to", help="Output topic, e.g. /model/prediction. Its "
                                 "messages must carry the input's header.stamp."),
    duration: float = typer.Option(30.0, "--duration", help="Seconds to listen."),
    deadline_ms: float = typer.Option(None, "--deadline-ms", help="Latency deadline."),
    same_machine: bool = typer.Option(False, "--same-machine", help="The sensor publishes "
                                      "on THIS machine (data age then needs no clock sync)."),
    source_bound_ms: float = typer.Option(None, "--source-bound-ms", help="The sensor "
                                          "machine's chrony error bound (from `edgelens "
                                          "selftest` there), for cross-machine data age."),
    save: str = typer.Option("edgelens_ros2.json", "--save", help="Output JSON path."),
):
    """Sensor-to-prediction latency from header timestamps: chain latency
    (no clock sync needed), data age, rate, jitter and gaps per topic."""
    from .hardware.telemetry import TelemetryRecorder
    from .ros2 import latency as rl
    from .ros2.node import Ros2Unavailable, record as ros_record
    console.print("[dim]edgelens ros2 is experimental in v0.2.[/dim]")
    rec = TelemetryRecorder(interval_s=0.2)
    rec.start()
    try:
        with console.status(f"Listening for {duration:.0f} s..."):
            src, dst, info = ros_record(from_topic, to_topic, duration)
    except Ros2Unavailable as e:
        rec.stop()
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    tel, series = rec.stop(), rec.series()
    summary, chain, ages = rl.summarize(src, dst, same_machine=same_machine,
                                        source_bound_ms=source_bound_ms)
    summary["subscription"] = info
    result = rl.to_result(summary, chain, ages, from_topic=from_topic, to_topic=to_topic,
                          deadline_ms=deadline_ms, telemetry_summary=tel,
                          telemetry_series=series)
    print_ros2(result)
    if result.get("iterations"):
        _print_benchmark(result)
    _save(result, save)


def print_ros2(result):
    r = result.get("ros2") or {}
    t = Table(title="ROS 2 topics")
    for col in ("Topic", "Messages", "Rate", "Interval jitter", "Gaps (missing)", "Header"):
        t.add_column(col)
    for side in ("from", "to"):
        st = r.get(side)
        if not st:
            continue
        t.add_row(st["topic"], str(st["messages"]),
                  f"{st.get('rate_hz')} Hz" if st.get("rate_hz") else "-",
                  f"{st.get('interval_jitter_ms')} ms" if "interval_jitter_ms" in st else "-",
                  f"{st.get('gaps', 0)} ({st.get('estimated_missing_messages', 0)})",
                  "yes" if st.get("has_header_stamp") else "no")
    console.print(t)
    ch, age = r.get("chain") or {}, r.get("data_age") or {}
    console.print(f"[bold]Chain latency:[/bold] {ch.get('matched', 0)} output messages "
                  f"matched by header.stamp ({ch.get('unmatched', 0)} unmatched). "
                  f"[dim]{ch.get('note', '')}[/dim]")
    a = age.get("at_output_ms") or age.get("from_ms") or {}
    unc = age.get("uncertainty_ms")
    console.print(f"[bold]Data age:[/bold] p50 {a.get('p50')} ms · p99 {a.get('p99')} ms · "
                  f"max {a.get('max')} ms · "
                  + (f"clock uncertainty ±{unc} ms" if unc is not None
                     else "[yellow]INCONCLUSIVE across machines[/yellow]")
                  + f" [dim]({age.get('uncertainty_basis')})[/dim]")
    console.print(f"[dim]Latency used for deadline/validate: {r.get('latency_basis')}[/dim]")


@app.command()
def include():
    """Print the directory containing edgelens_mark.h, for  -I$(edgelens include)."""
    typer.echo(INCLUDE_DIR)


__all__ = ["record", "watch", "import_log", "include", "print_integrity", "print_process"]
