"""
edgelens.cli_evidence
-----------------------
v0.2 Layer 3 commands — evidence and workflow:

    edgelens validate run.json --contract c.yaml [--evidence e.md]
    edgelens contract init | check | run        assumption contracts (E1)
    edgelens export run.json ... --perfetto t.json   one timeline (E2)

Registered on the main app by edgelens.cli.
"""

import json
import math
import time
from pathlib import Path
from typing import List

import typer
from rich.panel import Panel
from rich.table import Table

from . import contract as ct
from .cli import _print_integrity_issues, _sibling, _write_or_exit, app, console
from .validate.engine import exit_code as validate_exit_code

contract_app = typer.Typer(help="Assumption contracts: requirements AND the conditions they "
                                "are claimed under (docs/contracts.md).")
app.add_typer(contract_app, name="contract")

STYLE = {ct.PASS: "green", ct.FAIL: "red", ct.NOT_MEASURED: "yellow", ct.HOLDS: "green",
         ct.VIOLATED: "red", ct.NOT_VERIFIED: "yellow", ct.MET: "green", ct.NOT_MET: "red"}


def _st(status, reason=None):
    s = f"[{STYLE[status]}]{status.replace('_', ' ')}[/{STYLE[status]}]"
    return s + (f"\n[dim]{reason}[/dim]" if reason else "")


def _load_contract_or_exit(path):
    try:
        return ct.load(path)
    except ct.ContractError as e:
        console.print(f"[red]Contract error: {e}[/red]")
        raise typer.Exit(2)


def print_assumptions(rows, evidence=(), title="Assumptions (conditions of the claim)"):
    if not rows and not evidence:
        return
    t = Table(title=title)
    for col in ("ID", "Assumption", "Expected", "Measured", "Status"):
        t.add_column(col)
    for a in rows:
        t.add_row(a.get("id") or "", a["label"], ct._fmt(a["expected"]),
                  ct._fmt(a["measured"], a.get("unit")), _st(a["status"], a.get("reason")))
    for e in evidence:
        t.add_row("", e["label"], str(e["expected"]), ct._fmt(e["measured"], e.get("unit")),
                  _st(e["status"], e.get("reason")))
    console.print(t)


def print_contract_verdict(v):
    c = v["contract"]
    console.print(f"[bold]Contract[/bold] {c['name']}"
                  f"{' v' + str(c['version']) if c.get('version') else ''}"
                  f"{' · ' + c['requirements_doc'] if c.get('requirements_doc') else ''}"
                  f" [dim]({c.get('file')}, sha256 {str(c.get('sha256'))[:12]})[/dim]")
    print_assumptions(v["assumptions"], v["evidence_conditions"])
    t = Table(title="Guarantees")
    for col in ("ID", "Requirement", "Limit", "Measured", "Result"):
        t.add_column(col)
    for r in v["checks"]:
        op = "≤" if r["direction"] == "max" else "≥"
        t.add_row(r.get("id") or "", r["label"], f"{op} {ct._fmt(r['limit'], r['unit'])}",
                  ct._fmt(r["measured"], r["unit"]), _st(r["status"], r.get("reason")
                                                         or r.get("detail")))
    console.print(t)
    verdict = v["verdict"]
    if verdict == ct.PASS:
        console.print(Panel("Every assumption held and every guarantee was measured and met.",
                            title="✅ PASS", style="bold green", border_style="green"))
    elif verdict == ct.FAIL:
        d = (v.get("diagnosis") or {}).get("primary") or {}
        ls = v.get("limiting_stage")
        body = (f"[bold]Limiting stage:[/bold] {ls['stage']}\n\n" if ls else "")
        if d:
            body += (f"[bold]Why:[/bold] {d['type']} ({d['evidence_strength']} evidence)\n"
                     f"{d['detail']}\n\n[cyan]Next experiment → {d['recommendation']}[/cyan]")
        console.print(Panel(body or "A guarantee was not met.", title="❌ CONTRACT NOT MET",
                            border_style="red"))
    else:
        console.print(Panel("\n".join(v["notes"]) or "Inconclusive.", title="⚠️  INCONCLUSIVE",
                            style="yellow", border_style="yellow"))
    _print_integrity_issues([i for i in v.get("integrity") or [] if i["severity"] != "info"])


def validate_with_contract(path, data, contract_path, evidence=None):
    c, sha = _load_contract_or_exit(contract_path)
    v = ct.evaluate(data, c, sha)
    if data.get("mode") == "demo":
        console.print("[yellow]This is a --demo (simulated) result: it cannot validate "
                      "anything.[/yellow]")
    print_contract_verdict(v)
    out = _sibling(Path(path), "validation")
    _write_or_exit(out, json.dumps(v, indent=2, default=str))
    console.print(f"[green]Saved →[/green] {out}")
    if evidence:
        _write_or_exit(evidence, ct.evidence_markdown(v, str(path)))
        console.print(f"[green]Evidence →[/green] {evidence}")
    return validate_exit_code(v["verdict"])


# ---------------------------------------------------------------- contract init / check

@contract_app.command("init")
def contract_init(
    path: str = typer.Argument("edgelens.contract.yaml", help="Where to write the template."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing file."),
):
    """Write a commented contract template to start from."""
    if Path(path).exists() and not force:
        console.print(f"[red]{path} exists (use --force to overwrite).[/red]")
        raise typer.Exit(1)
    _write_or_exit(path, ct.TEMPLATE)
    console.print(f"[green]Wrote[/green] {path} — edit it, then: edgelens contract check {path}")


@contract_app.command("check")
def contract_check(path: str = typer.Argument(..., help="Contract file (.yaml / .json).")):
    """Check a contract file: unknown keys (typos), types, missing parts. Prints how a
    run must be made to satisfy it."""
    c, sha = _load_contract_or_exit(path)
    t = Table(title=f"{c['contract']} {('v' + str(c['version'])) if c.get('version') else ''}")
    t.add_column("Section")
    t.add_column("Key")
    t.add_column("Value")
    t.add_column("ID")
    for k, s in c["assumes"].items():
        extra = f" ± {s['tolerance_pct']:g}%" if "tolerance_pct" in s else ""
        t.add_row("assumes", k, f"{ct._fmt(s['value'])}{extra}", s.get("id") or "")
    for k, s in c["guarantees"].items():
        extra = f" (deadline {s['deadline_ms']:g} ms)" if s.get("deadline_ms") is not None else ""
        t.add_row("guarantees", k, f"{s['value']:g}{extra}", s.get("id") or "")
    if c["min_duration_s"] is not None:
        t.add_row("evidence", "min_duration", f"{c['min_duration_s']:g} s", "")
    if c["min_iterations"] is not None:
        t.add_row("evidence", "min_iterations", str(c["min_iterations"]), "")
    console.print(t)
    p = plan(c)
    for w in p["warnings"]:
        console.print(f"[yellow]! {w}[/yellow]")
    console.print(f"[green]OK[/green] sha256 {sha[:12]} · a run that satisfies it: "
                  f"[bold]{p['describe']}[/bold]\n[dim]edgelens contract run {path} "
                  f"--model M.onnx | --pipeline P.py[/dim]")


# ---------------------------------------------------------------- contract run

SOAK_KEYS = ("max_latency_drift_pct", "max_mem_growth_mb_per_h", "max_throttled_windows")
ENERGY_KEYS = ("max_energy_mj", "max_power_w", "max_energy_per_period_mj")
OUTPUT_KEYS = ("max_output_error", "min_decision_agreement")
_PCT = {"p50_ms": 0.5, "p95_ms": 0.95, "p99_ms": 0.99, "p99_9_ms": 0.999}


def plan(c):
    """How to measure so that the run can satisfy the contract's evidence
    conditions: kind of run, period, iterations / duration, deadline."""
    from .core.stats import min_samples_for
    g, a = c["guarantees"], c["assumes"]
    warnings = []
    period = a["input_period_ms"]["value"] if "input_period_ms" in a else None
    deadline = (g.get("max_miss_ratio") or {}).get("deadline_ms") or period
    if "max_miss_ratio" in g and deadline is None:
        warnings.append("max_miss_ratio needs a deadline: give it one (deadline_ms) or "
                        "assume an input_period_ms")
    need_n = max([c["min_iterations"] or 1] +
                 [min_samples_for(_PCT[k]) for k in g if k in _PCT])
    if any(k in g for k in ENERGY_KEYS):
        need_n = max(need_n, 30)
    kind = "benchmark"
    if any(k in g for k in SOAK_KEYS):
        kind = "soak"
        if c["min_duration_s"] is None:
            warnings.append("soak guarantees (drift / memory growth / throttling) need a "
                            "min_duration, e.g. 30m")
        if "max_energy_per_period_mj" in g:
            warnings.append("max_energy_per_period_mj comes from `edgelens periodic`, not a "
                            "soak: it will be NOT MEASURED; split it into its own contract")
    elif "max_energy_per_period_mj" in g:
        kind = "periodic"
        if period is None:
            warnings.append("max_energy_per_period_mj needs assumes.input_period_ms")
    if any(k in g for k in OUTPUT_KEYS):
        warnings.append("output guarantees come from `edgelens sweep`: run it and validate "
                        "the backend file (e.g. sweep.tensorrt-fp16.json) with --contract")
    duration = c["min_duration_s"]
    iterations, max_seconds = need_n, None
    if kind == "soak":
        window = min(60.0, max(5.0, (duration or 600) / 10.0))
        desc = (f"soak for {duration or 0:g} s (windows of {window:g} s)"
                + (f", one input every {period:g} ms" if period else ", back to back"))
        return {"kind": kind, "period_ms": period, "deadline_ms": deadline,
                "duration_s": duration, "window_s": window, "iterations": None,
                "max_seconds": None, "warnings": warnings, "describe": desc,
                "idle_baseline_s": 0.0}
    if period:
        if duration:
            iterations = max(iterations, int(math.ceil(duration * 1000.0 / period)) + 1)
        est = iterations * period / 1000.0
        desc = (f"{kind}: {iterations} inputs, one every {period:g} ms "
                f"(about {est / 60:.1f} min)")
    else:
        if duration:
            iterations = 10 ** 9            # stop on time, at least need_n
            max_seconds = duration
            desc = f"{kind}: back to back for {duration:g} s (>= {need_n} iterations)"
        else:
            desc = f"{kind}: {iterations} iterations back to back"
    idle = 10.0 if any(k in g for k in ENERGY_KEYS) else 0.0
    return {"kind": kind, "period_ms": period, "deadline_ms": deadline, "duration_s": duration,
            "iterations": iterations, "min_iterations": need_n, "max_seconds": max_seconds,
            "warnings": warnings, "describe": desc, "idle_baseline_s": idle}


PRE_RUN = ("board", "power_mode", "clocks_pinned", "max_start_temp_c", "max_background_cpu_pct",
           "provider")


def preflight(c, pipe_provider=None, seconds=2.0):
    """Assumptions that can be checked BEFORE spending 30 minutes on a run."""
    from .core.identity import capture_environment
    from .hardware.telemetry import TelemetryRecorder
    keys = [k for k in PRE_RUN if k in c["assumes"]]
    if not keys:
        return []
    fake = {"environment": capture_environment(), "telemetry": {}}
    if "max_start_temp_c" in keys or "max_background_cpu_pct" in keys:
        rec = TelemetryRecorder(interval_s=0.5, use_tegrastats=False, use_power=False)
        rec.start()
        time.sleep(seconds)
        fake["telemetry"] = rec.stop()
    if pipe_provider:
        fake["pipeline"] = {"config": {"provider": pipe_provider}}
    return [ct.check_assumption(fake, k, c["assumes"][k]) for k in keys
            if k != "provider" or pipe_provider]


@contract_app.command("run")
def contract_run(
    contract_file: str = typer.Argument(..., help="Contract file (.yaml / .json)."),
    model: str = typer.Option(None, "--model", help="ONNX model."),
    pipeline: str = typer.Option(None, "--pipeline", help="Pipeline script (build_pipeline)."),
    provider: str = typer.Option(None, "--provider", help="ONNX Runtime provider (--model)."),
    input_shape: List[str] = typer.Option(None, "--input-shape", help="Dynamic input shape."),
    save: str = typer.Option("edgelens_contract_run.json", "--save", help="Result file."),
    evidence: str = typer.Option(None, "--evidence", help="Markdown evidence record "
                                 "(default: <save>.evidence.md)."),
    perfetto: str = typer.Option(None, "--perfetto", help="Also export the run's timeline "
                                 "(Perfetto JSON) to this file."),
    run_anyway: bool = typer.Option(False, "--run-anyway", help="Measure even when the "
                                    "pre-run check finds a violated assumption (the verdict "
                                    "will be INCONCLUSIVE)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only print the plan and the "
                                 "pre-run check."),
):
    """Measure exactly what the contract needs (period, iterations, duration, deadline,
    idle baseline), then validate against it. Checks the assumptions it can BEFORE the
    run, so a CI job doesn't spend 30 minutes on a run that can only be INCONCLUSIVE.

    Exit codes: 0 PASS, 1 FAIL, 2 INCONCLUSIVE or a violated pre-run assumption."""
    from .cli_insights import _build_pipe
    c, sha = _load_contract_or_exit(contract_file)
    p = plan(c)
    console.print(f"[bold]Plan:[/bold] {p['describe']}")
    for w in p["warnings"]:
        console.print(f"[yellow]! {w}[/yellow]")
    pre = preflight(c, pipe_provider=provider)
    if pre:
        print_assumptions(pre, title="Pre-run check")
    bad = [r for r in pre if r["status"] == ct.VIOLATED]
    if dry_run:
        raise typer.Exit(2 if bad else 0)
    stem = save[:-5] if save.endswith(".json") else save
    evidence = evidence or f"{stem}.evidence.md"
    if bad and not run_anyway:
        # still leave an evidence record: a CI job summary must say WHY nothing ran
        from .core.identity import capture_environment
        env = capture_environment()
        pm = (env.get("power_mode") or {}).get("name")
        v = {"verdict": ct.INCONCLUSIVE, "guarantees_verdict": None, "checks": [],
             "assumptions": pre, "evidence_conditions": [], "integrity": [],
             "notes": ["Not measured: an assumption was already violated before the run."]
             + [f"Assumption {r['assumption']}{' [' + r['id'] + ']' if r.get('id') else ''} "
                f"VIOLATED: {r.get('reason') or ''}" for r in bad],
             "run": {"board": env.get("board"), "power_mode": pm},
             "contract": {"name": c["contract"], "version": c.get("version"),
                          "file": c.get("file"), "sha256": sha,
                          "requirements_doc": c.get("requirements_doc"),
                          "description": c.get("description")}}
        _write_or_exit(evidence, ct.evidence_markdown(v, None))
        console.print(Panel("\n".join(f"{r['assumption']}: {r.get('reason') or ''}" for r in bad)
                            + "\n\nFix the setup (or pass --run-anyway to measure anyway; the "
                              "verdict can only be INCONCLUSIVE).",
                            title="⚠️  ASSUMPTION VIOLATED BEFORE THE RUN", style="yellow",
                            border_style="yellow"))
        raise typer.Exit(2)

    pipe, source, model_path = _build_pipe(model, pipeline, provider, input_shape)
    try:
        if p["kind"] == "soak":
            from . import soak as sk
            if not p["duration_s"]:
                console.print("[red]This contract needs a min_duration for a soak.[/red]")
                raise typer.Exit(2)
            result = sk.run_soak_pipeline(pipe, p["duration_s"], window_s=p["window_s"],
                                          period_ms=p["period_ms"], deadline_ms=p["deadline_ms"],
                                          jsonl_path=f"{stem}.jsonl", pipeline_source=source,
                                          model_path=model_path,
                                          progress=lambda m: console.print(f"[dim]{m}[/dim]"))
        elif p["kind"] == "periodic":
            from . import periodic as per
            with console.status("Measuring (periodic)...") as status:
                result = per.run_periodic(pipe, [p["period_ms"]], iterations=p["iterations"],
                                          warm_iterations=min(200, max(20, p["iterations"])),
                                          idle_baseline_s=p["idle_baseline_s"],
                                          deadline_ms=p["deadline_ms"], pipeline_source=source,
                                          model_path=model_path,
                                          progress=lambda m: status.update(m))
        else:
            from .core.harness import run_harness
            with console.status(f"Measuring: {p['describe']} ..."):
                result = run_harness(pipe, iterations=p["iterations"], warmup=10,
                                     source=getattr(pipe, "source", None),
                                     deadline_ms=p["deadline_ms"], period_ms=p["period_ms"],
                                     pace=bool(p["period_ms"]),
                                     idle_baseline_s=p["idle_baseline_s"] or None,
                                     pipeline_source=source, model_path=model_path,
                                     max_seconds=p["max_seconds"])
    except (RuntimeError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    result.setdefault("contract_run", {}).update(plan={k: v for k, v in p.items()
                                                       if k != "warnings"},
                                                 contract=c["contract"], contract_sha256=sha)
    _write_or_exit(save, json.dumps(result, indent=2, default=str))
    console.print(f"[green]Result →[/green] {save}")
    if perfetto:
        from .timeline import TimelineError, export
        try:
            info = export([save], perfetto)
            console.print(f"[green]Timeline →[/green] {perfetto} ({info['spans']} spans)")
        except TimelineError as e:
            console.print(f"[yellow]No timeline: {e}[/yellow]")
    code = validate_with_contract(save, result, contract_file, evidence)
    raise typer.Exit(code)


# ---------------------------------------------------------------- export (E2)

@app.command("export")
def export_cmd(
    files: List[str] = typer.Argument(..., help="Result files (benchmark, record, watch, "
                                      "periodic, sweep backend files...). Several files "
                                      "recorded at the same time share one timeline."),
    perfetto: str = typer.Option(None, "--perfetto", help="Write a Perfetto / Chrome JSON "
                                 "trace here. Open it at https://ui.perfetto.dev."),
    align: str = typer.Option("auto", "--align", help="auto: same boot -> real monotonic "
                              "time, else each file from its start | mono | start."),
):
    """One timeline: spans of every process, end-to-end iterations, deadline misses, and
    board counters (CPU/GPU load, GPU clock, temperature, power and every rail)."""
    from .timeline import TimelineError, export
    if not perfetto:
        console.print("[red]Choose a format: --perfetto timeline.json[/red]")
        raise typer.Exit(2)
    if align not in ("auto", "mono", "start"):
        console.print("[red]--align is auto, mono or start.[/red]")
        raise typer.Exit(2)
    for f in files:
        if not Path(f).exists():
            console.print(f"[red]No such file: {f}[/red]")
            raise typer.Exit(1)
    try:
        info = export(files, perfetto, align)
    except TimelineError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except OSError as e:
        console.print(f"[red]Cannot write {perfetto}: {e.strerror or e}[/red]")
        raise typer.Exit(1)
    for n in info["notes"]:
        console.print(f"[yellow]! {n}[/yellow]")
    console.print(f"[green]Timeline →[/green] {perfetto}: {info['spans']} spans, "
                  f"{info['counter_samples']} counter samples (time base: {info['alignment']}).\n"
                  f"Open https://ui.perfetto.dev → 'Open trace file' (it stays in your "
                  f"browser; nothing is uploaded).")
