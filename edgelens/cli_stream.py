"""
edgelens.cli_stream
---------------------
v0.2 Layer 4 commands — windowed time-series pipelines (edgelens.stream):

    edgelens stream init | check | fit-scaler
    edgelens replay   spec.yaml data/*.csv [--speed 1] [--save run.json]
    edgelens detect   replays/*.json [--thresholds 0.9,0.7,0.5] [--mean-steps 20]
    edgelens envelope fit | check
    edgelens perturb  spec.yaml data.csv --faults faults.yaml
    edgelens parity   spec.yaml data.csv [--reference x.bin]

Registered on the main app by edgelens.cli. docs/stream.md.
"""

import copy
import json
from pathlib import Path
from typing import List

import typer
from rich.table import Table

from .cli import _load_result_or_exit, _write_or_exit, app, console

stream_app = typer.Typer(help="Stream specs: a windowed time-series pipeline exactly as "
                              "trained (docs/stream.md).")
app.add_typer(stream_app, name="stream")
envelope_app = typer.Typer(help="The training envelope of a model's inputs, and a runtime "
                                "guard against inputs it never saw.")
app.add_typer(envelope_app, name="envelope")


# ---------------------------------------------------------------- helpers

def _spec_or_exit(path, ignore_scaler=False):
    from .stream.spec import SpecError, load_spec
    try:
        return load_spec(path, ignore_scaler=ignore_scaler)[0]
    except SpecError as e:
        console.print(f"[red]Spec error: {e}[/red]")
        raise typer.Exit(2)


def _recordings_or_exit(paths, spec):
    from .stream.data import load_csv
    recs = []
    for p in paths:
        try:
            recs.append(load_csv(p, spec))
        except (OSError, ValueError) as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(2)
    if not recs:
        console.print("[red]No recordings given.[/red]")
        raise typer.Exit(2)
    return recs


def _model_or_exit(spec):
    from .stream.model import load_model
    from .stream.spec import SpecError
    try:
        return load_model(spec)
    except SpecError as e:
        console.print(f"[red]Model: {e}[/red]")
        raise typer.Exit(2)


def _guard_or_exit(spec, path=None):
    from .stream import envelope as envmod
    p = path or spec["envelope"]["file"]
    if not p:
        return None
    try:
        return envmod.Guard(envmod.load(p, spec), spec)
    except (OSError, ValueError) as e:
        console.print(f"[red]Envelope: {e}[/red]")
        raise typer.Exit(2)


def _s(v, unit="s", d=1):
    if v is None:
        return "—"
    return f"{v:.{d}f} {unit}".strip()


def _floats(text):
    try:
        return [float(x) for x in text.split(",") if x.strip()]
    except ValueError:
        console.print(f"[red]Expected numbers separated by commas, got '{text}'.[/red]")
        raise typer.Exit(2)


# ---------------------------------------------------------------- stream

@stream_app.command("init")
def stream_init(path: str = typer.Argument("pipeline.stream.yaml")):
    """Write a commented stream spec template."""
    from .stream.spec import TEMPLATE
    if Path(path).exists():
        console.print(f"[red]{path} exists; not overwriting.[/red]")
        raise typer.Exit(1)
    _write_or_exit(path, TEMPLATE)
    console.print(f"[green]Wrote[/green] {path} — edit it, then `edgelens stream check {path}`")


@stream_app.command("check")
def stream_check(spec_path: str = typer.Argument(..., help="Stream spec (.yaml / .json)."),
                 data: List[str] = typer.Argument(None, help="Recordings to audit (CSV)."),
                 model: bool = typer.Option(True, "--model/--no-model",
                                            help="Load the model and check its input shape.")):
    """Validate a spec, say what it implies (history, warm-up, smoothing delay),
    and audit recordings (rate, gaps, dead columns)."""
    spec = _spec_or_exit(spec_path)
    from .stream.spec import describe
    console.print(f"[bold]{spec['name']}[/bold] [dim]({spec['file']}, sha256 "
                  f"{spec['sha256'][:12]})[/dim]")
    for line in describe(spec):
        console.print(f"  • {line}")
    if model and (spec["model"]["onnx"] or spec["model"]["python"]):
        m = _model_or_exit(spec)
        console.print(f"  • model: {m.info.get('file') or m.info.get('ref')} on "
                      f"{m.info.get('provider')} — input shape OK")
    if data:
        from .stream.data import audit, find_constant_columns
        recs = _recordings_or_exit(data, spec)
        const = find_constant_columns(recs)
        t = Table(title="Recordings")
        for c in ("File", "Rows", "Duration", "Rate", "Gaps", "Truth"):
            t.add_column(c)
        for r in recs:
            a = audit(r, spec)
            t.add_row(Path(r.source).name, str(a["rows"]), _s(a["duration_s"], "s", 0),
                      f"{a['rate_hz_measured']} Hz" + (" [red]≠ spec[/red]"
                                                       if a["rate_mismatch"] else ""),
                      (f"[yellow]{len(a['gaps'])} ({a['gap_time_s']:g} s, longest "
                       f"{max(g['length_s'] for g in a['gaps']):g} s)[/yellow]"
                       if a["gaps"] else "0"),
                      "yes" if r.truth is not None else "—")
        console.print(t)
        if const:
            console.print(f"[yellow]{len(const)} of {len(spec['data']['columns'])} columns never "
                          f"change in these recordings:[/yellow] {', '.join(const)}")


@stream_app.command("fit-scaler")
def stream_fit_scaler(spec_path: str = typer.Argument(...),
                      data: List[str] = typer.Argument(..., help="TRAINING recordings only."),
                      kind: str = typer.Option("robust", "--kind",
                                               help="robust (median/IQR), standard, minmax"),
                      out: str = typer.Option("scaler.json", "--out")):
    """Fit the per-column scaler on training recordings, as scikit-learn would
    (RobustScaler / StandardScaler / MinMaxScaler defaults), and save it as JSON.

    Have the training code's own scaler? Export it instead:
    json.dump({"center": s.center_.tolist(), "scale": s.scale_.tolist(), "columns": cols}, f)"""
    spec = _spec_or_exit(spec_path, ignore_scaler=True)
    from .stream.data import find_constant_columns, fit_scaler
    recs = _recordings_or_exit(data, spec)
    try:
        sc = fit_scaler(recs, kind)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2)
    sc["constant_columns"] = find_constant_columns(recs)
    _write_or_exit(out, json.dumps(sc, indent=1))
    console.print(f"[green]Saved →[/green] {out} ({kind}, {sc['rows']} rows, "
                  f"{len(recs)} recordings). Name it in the spec: scaler: {{file: {out}}}")
    if sc["constant_columns"]:
        console.print(f"[yellow]Never change in training ({len(sc['constant_columns'])}):"
                      f"[/yellow] {', '.join(sc['constant_columns'])}")


# ---------------------------------------------------------------- replay

def _replay_row(t, res):
    d = res["decision"]
    st = res["stage_stats_ms"]
    o, r = d.get("output") or {}, d.get("raw") or {}
    t.add_row(res["stream"]["recording"]["file"], str(res["iterations"]),
              _s((st.get("features") or {}).get("p50"), "ms", 3),
              _s((st.get("inference") or {}).get("p50"), "ms", 3),
              _s((res.get("service_latency") or {}).get("p99"), "ms", 3),
              _s(r.get("time_to_detect_s")), _s(o.get("time_to_detect_s")),
              "[red]MISSED[/red]" if o.get("missed") else str(o.get("false_alarms", "—")),
              "—" if d.get("in_envelope_pct") is None else f"{d['in_envelope_pct']:g}%")


def _replay_table(title):
    t = Table(title=title)
    for c in ("Recording", "Steps", "Features p50", "Inference p50", "Step p99",
              "Detect (raw)", "Detect (output)", "False alarms", "In envelope"):
        t.add_column(c)
    return t


@app.command()
def replay(spec_path: str = typer.Argument(..., help="Stream spec."),
           data: List[str] = typer.Argument(..., help="Recordings (CSV)."),
           save: str = typer.Option(None, "--save", help="Result file (one recording)."),
           save_dir: str = typer.Option(None, "--save-dir",
                                        help="One result per recording in this folder."),
           speed: float = typer.Option(None, "--speed", help="1 = real time (one step per "
                                       "trained step), 10 = ten times faster. Default: as "
                                       "fast as possible."),
           features: str = typer.Option("incremental", "--features",
                                        help="incremental (one new block per step) or "
                                             "recompute (every block of the window, each "
                                             "step: a direct port of training code)"),
           deadline_ms: float = typer.Option(None, "--deadline-ms"),
           max_steps: int = typer.Option(None, "--max-steps"),
           telemetry: bool = typer.Option(True, "--telemetry/--no-telemetry")):
    """Replay recordings through the spec's pipeline, one block at a time:
    per-stage timing, board telemetry, and every decision against ground truth."""
    spec = _spec_or_exit(spec_path)
    if features not in ("incremental", "recompute"):
        console.print("[red]--features: incremental or recompute[/red]")
        raise typer.Exit(2)
    if save and len(data) > 1:
        console.print("[red]--save is for one recording; use --save-dir for several.[/red]")
        raise typer.Exit(2)
    recs = _recordings_or_exit(data, spec)
    model = _model_or_exit(spec)
    _guard_or_exit(spec)                       # fail early on a bad envelope file
    from .stream.runner import replay as run_replay
    dead = None
    sc_file = spec["scaler"].get("file")
    if sc_file:
        try:
            with open(Path(spec["base_dir"]) / sc_file) as f:
                dead = json.load(f).get("constant_columns")
        except (OSError, ValueError):
            dead = None
    t = _replay_table(f"Replay — {spec['name']} ({features} features"
                      f"{', real time x' + format(speed, 'g') if speed else ', unpaced'})")
    results = []
    for r in recs:
        console.print(f"[dim]{Path(r.source).name}: {len(r) // spec['features']['block']} "
                      f"steps...[/dim]")
        try:
            res = run_replay(r, spec, model, mode=features, speed=speed,
                             deadline_ms=deadline_ms, telemetry=telemetry, max_steps=max_steps,
                             train_constant=dead)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(2)
        results.append(res)
        _replay_row(t, res)
        out = save or (str(Path(save_dir) / (r.name + ".json")) if save_dir else None)
        if out:
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
            _write_or_exit(out, json.dumps(res, indent=1))
    console.print(t)
    _print_budget(results[0]["decision"].get("budget"), results[0])
    if save:
        console.print(f"[green]Saved →[/green] {save}  (next: edgelens detect {save})")
    elif save_dir:
        console.print(f"[green]Saved →[/green] {save_dir}/  (next: edgelens detect "
                      f"{save_dir}/*.json --save detect.json)")
    if len(results) > 1:
        from .stream.detect import aggregate
        _print_aggregate(aggregate([x["decision"] for x in results]))


def _print_budget(b, res=None):
    if not b:
        return
    console.print("[bold]Where decision time goes[/bold] (built into the pipeline):")
    rows = [("history the model sees", b.get("history_s"), "s"),
            ("warm-up (padded window)", b.get("warmup_s"), "s"),
            ("wait for a block to fill (max)", b.get("block_wait_max_s"), "s"),
            ("output mean (approx. delay)", b.get("mean_delay_s"), "s"),
            ("slew limit: fastest move to the threshold", b.get("slew_floor_s"), "s"),
            ("post-processing cost (output vs raw)", b.get("post_processing_cost_s"), "s"),
            ("compute per step (p99)", b.get("compute_p99_ms"), "ms")]
    for label, v, u in rows:
        if v is not None:
            console.print(f"  {label:44s} {v:g} {u}")


def _print_aggregate(agg):
    t = Table(title=f"All recordings ({agg['recordings']})")
    for c in ("Output", "Events", "Missed", "Time to detect (max / median)", "False alarms",
              "False alarms / h", "Early alarm", "RMSE (mean)"):
        t.add_column(c)
    for part in ("raw", "output"):
        p = agg.get(part) or {}
        ttd = p.get("time_to_detect_s")
        t.add_row(part, str(p.get("events")), str(p.get("missed")),
                  f"{ttd['max']:.1f} / {ttd['median']:.1f} s" if ttd else "—",
                  str(p.get("false_alarms")), str(p.get("false_alarms_per_h")),
                  _s(p.get("early_alarm_s"), "s", 0), str(p.get("rmse_mean")))
    console.print(t)
    if agg.get("in_envelope_pct_min") is not None:
        console.print(f"Inputs inside the training envelope: at least "
                      f"{agg['in_envelope_pct_min']:g}% of the predictions in every recording")


# ---------------------------------------------------------------- detect

@app.command()
def detect(results: List[str] = typer.Argument(..., help="Replay results (JSON)."),
           threshold: float = typer.Option(None, "--threshold", help="Alarm threshold "
                                           "(default: the spec's)."),
           thresholds: str = typer.Option(None, "--thresholds", help="Decision-latency curve: "
                                          "time to detect at each threshold, e.g. 0.9,0.7,0.5"),
           slew_per_step: float = typer.Option(None, "--slew-per-step", help="What-if: another "
                                               "slew limit (0 = none)."),
           mean_steps: int = typer.Option(None, "--mean-steps", help="What-if: another output "
                                          "mean length (1 = none)."),
           save: str = typer.Option(None, "--save", help="Write the detection result (JSON; "
                                    "`edgelens validate --contract` reads it)."),
           markdown: str = typer.Option(None, "--markdown", help="Write a Markdown report.")):
    """Decision latency and alarm quality over replayed recordings: time from
    event to alarm, false alarms per healthy hour, missed events — for the raw
    model output and after post-processing. Recomputes from the stored outputs,
    so thresholds and smoothing can be tuned without re-running the model."""
    import numpy as np

    from .stream import detect as det
    from .stream.features import filter_series
    docs = []
    for p in results:
        d = _load_result_or_exit(p)
        if not (d.get("decision") or {}).get("series"):
            console.print(f"[red]{p} has no decision series: it is not an `edgelens replay` "
                          f"result.[/red]")
            raise typer.Exit(2)
        docs.append((p, d))
    rule = copy.deepcopy(docs[0][1]["stream"].get("decision_rule"))
    if not rule:
        console.print("[red]The replayed spec has no decision rule (decision: threshold).[/red]")
        raise typer.Exit(2)
    if threshold is not None:
        rule["threshold"] = threshold
        rule["truth_threshold"] = threshold
    st0 = docs[0][1]["stream"]
    step = st0["trained_step_s"]
    orig = st0.get("output_rule") or {"slew_per_step": None, "mean_steps": 1}
    whatif = slew_per_step is not None or mean_steps is not None
    out_rule = dict(orig)
    if slew_per_step is not None:
        out_rule["slew_per_step"] = slew_per_step or None
    if mean_steps is not None:
        out_rule["mean_steps"] = max(1, mean_steps)
    spec_like = {"features": {"step_s": step,
                              "sequence": st0.get("sequence") or
                              docs[0][1]["pipeline"]["config"]["window"][0],
                              "padding": st0.get("padding", "zeros")},
                 "output": out_rule, "decision": rule}
    per = []
    for p, d in docs:
        s = d["decision"]["series"]
        t = np.asarray(s["t_s"], dtype=np.float64)
        truth = np.asarray(s["truth"], dtype=np.float64) if s.get("truth") else None
        raw = np.asarray([np.nan if v is None else v for v in s["raw"]], dtype=np.float64)
        out = np.asarray([np.nan if v is None else v for v in s["out"]], dtype=np.float64)
        if whatif:
            ok = ~np.isnan(raw)
            out = np.full(len(raw), np.nan)
            out[ok] = filter_series(raw[ok], spec_like)
        summ = det.summarize(t, truth, raw, out, s["state"], spec_like,
                             compute_ms=(d.get("service_latency") or {}).get("p99"))
        summ["recording"] = d["stream"]["recording"]["file"]
        summ["result_file"] = p
        if thresholds:
            summ["curve"] = []
            for thr in _floats(thresholds):
                r2 = dict(rule, threshold=thr, truth_threshold=thr)
                summ["curve"].append({"threshold": thr,
                                      "raw": det.score(t, truth, raw, r2, step),
                                      "output": det.score(t, truth, out, r2, step)})
        per.append(summ)
    agg = det.aggregate(per)
    _print_detect(per, agg, rule, whatif, out_rule)
    if thresholds:
        _print_curve(per)
    doc = _detect_doc(docs, per, agg, rule, out_rule if whatif else None)
    if save:
        _write_or_exit(save, json.dumps(doc, indent=1))
        console.print(f"[green]Saved →[/green] {save}  (edgelens validate {save} --contract "
                      f"c.yaml)")
    if markdown:
        _write_or_exit(markdown, detect_markdown(doc))
        console.print(f"[green]Report →[/green] {markdown}")


def _print_detect(per, agg, rule, whatif, out_rule):
    op = "≤" if rule["alarm_when"] == "below" else "≥"
    title = f"Decisions — alarm when output {op} {rule['threshold']:g}"
    if whatif:
        title += (f" (what-if: slew {out_rule['slew_per_step'] or 'none'}, mean "
                  f"{out_rule['mean_steps']})")
    t = Table(title=title)
    for c in ("Recording", "Onset", "Event", "Alarm (raw)", "Alarm (output)",
              "Time to detect raw / output", "False alarms", "Early alarm", "RMSE"):
        t.add_column(c)
    for s in per:
        r, o = s.get("raw") or {}, s.get("output") or {}
        t.add_row(s["recording"], _s(o.get("onset_s"), "s", 0), _s(o.get("event_s"), "s", 0),
                  _s(r.get("alarm_after_onset_s"), "s", 0),
                  _s(o.get("alarm_after_onset_s"), "s", 0),
                  f"{_s(r.get('time_to_detect_s'))} / "
                  + ("[red]MISSED[/red]" if o.get("missed") else _s(o.get("time_to_detect_s"))),
                  str(o.get("false_alarms", "—")), _s(o.get("early_alarm_s"), "s", 0),
                  str(o.get("rmse")))
    console.print(t)
    _print_aggregate(agg)


def _print_curve(per):
    thr = [c["threshold"] for c in per[0]["curve"]]
    t = Table(title="Decision-latency curve: time to detect (output) per threshold")
    t.add_column("Recording")
    for x in thr:
        t.add_column(f"{x:g}")
    for s in per:
        cells = []
        for c in s["curve"]:
            o = c["output"]
            cells.append("[red]missed[/red]" if o.get("missed") else _s(o.get("time_to_detect_s"),
                                                                         "s", 0))
        t.add_row(s["recording"], *cells)
    console.print(t)


def _detect_doc(docs, per, agg, rule, whatif):
    from .core.stats import latency_stats
    base = copy.deepcopy(docs[0][1])
    lat = []
    for _, d in docs:
        lat.extend(d.get("latency_series_ms") or [])
    base.pop("latency_series_ms", None)
    base["trace"] = {"events": [], "truncated": False, "event_count": 0}
    base["telemetry_series"] = []
    base["mode"] = "detect"
    base["iterations"] = sum(d.get("iterations") or 0 for _, d in docs)
    if lat:
        from .core.stats import deadline_stats
        base["latency"] = latency_stats(lat)
        dl = (base.get("deadline") or {}).get("deadline_ms")
        if dl:
            base["deadline"] = deadline_stats(lat, dl)
    base["decision"] = {"rule": rule,
                        "what_if": whatif,
                        "aggregate": agg,
                        "recordings": [{k: v for k, v in s.items() if k != "series"}
                                       for s in per]}
    base["detect"] = {"files": [p for p, _ in docs]}
    return base


def detect_markdown(doc):
    dec = doc["decision"]
    rule = dec["rule"]
    agg = dec["aggregate"]
    op = "≤" if rule["alarm_when"] == "below" else "≥"
    L = [f"# Decision latency — {(doc.get('stream') or {}).get('spec', {}).get('name')}", "",
         f"Alarm when the output {op} {rule['threshold']:g}; event = ground truth {op} "
         f"{rule['truth_threshold']:g}; healthy = truth at {rule.get('healthy_level')}."
         + (f" What-if post-processing: {dec['what_if']}." if dec.get("what_if") else ""), "",
         "| Recording | Onset | Event | Time to detect (raw) | Time to detect (output) | "
         "False alarms | Early alarm | RMSE |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in dec["recordings"]:
        r, o = s.get("raw") or {}, s.get("output") or {}
        L.append(f"| {s['recording']} | {_s(o.get('onset_s'), 's', 0)} | "
                 f"{_s(o.get('event_s'), 's', 0)} | {_s(r.get('time_to_detect_s'))} | "
                 + ("MISSED" if o.get("missed") else _s(o.get("time_to_detect_s")))
                 + f" | {o.get('false_alarms', '—')} | {_s(o.get('early_alarm_s'), 's', 0)} | "
                   f"{o.get('rmse')} |")
    o = agg.get("output") or {}
    ttd = o.get("time_to_detect_s") or {}
    L += ["", f"**Worst time to detect: {_s(ttd.get('max'))}** (median "
          f"{_s(ttd.get('median'))}) · missed {o.get('missed')}/{o.get('events')} · "
          f"false alarms {o.get('false_alarms')} in {o.get('healthy_hours')} healthy hours · "
          f"compute per step p99 {_s((doc.get('latency') or {}).get('p99'), 'ms', 3)}", ""]
    b = (dec["recordings"][0].get("budget") or {}) if dec["recordings"] else {}
    if b:
        L.append("Built-in delays: " + ", ".join(
            f"{k.replace('_', ' ')} {v:g}" for k, v in b.items()
            if isinstance(v, (int, float)) and k != "slew_from_level"))
        L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- envelope

@envelope_app.command("fit")
def envelope_fit(spec_path: str = typer.Argument(...),
                 data: List[str] = typer.Argument(..., help="TRAINING recordings."),
                 out: str = typer.Option("envelope.json", "--out"),
                 quantile: float = typer.Option(0.001, "--quantile",
                                                help="Ignore this fraction at each end."),
                 margin: float = typer.Option(0.1, "--margin",
                                              help="Widen each range by this fraction.")):
    """Fit the per-input envelope of the training data (model inputs, scaled)."""
    spec = _spec_or_exit(spec_path)
    from .stream.envelope import fit
    recs = _recordings_or_exit(data, spec)
    env = fit(recs, spec, quantile=quantile, margin=margin)
    _write_or_exit(out, json.dumps(env, indent=1))
    console.print(f"[green]Saved →[/green] {out} ({env['steps']} steps, "
                  f"{len(env['features'])} inputs, {len(env['constant_features'])} constant in "
                  f"training). Name it in the spec: envelope: {{file: {out}}}")


@envelope_app.command("check")
def envelope_check(spec_path: str = typer.Argument(...),
                   data: List[str] = typer.Argument(...),
                   envelope: str = typer.Option(None, "--envelope",
                                                help="Default: the spec's envelope file."),
                   save: str = typer.Option(None, "--save")):
    """Which recordings, and which inputs, leave the training envelope."""
    spec = _spec_or_exit(spec_path)
    from .stream import envelope as envmod
    p = envelope or spec["envelope"]["file"]
    if not p:
        console.print("[red]No envelope: pass --envelope or name one in the spec.[/red]")
        raise typer.Exit(2)
    try:
        env = envmod.load(p, spec)
    except (OSError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2)
    recs = _recordings_or_exit(data, spec)
    rep = envmod.check(recs, spec, env)
    t = Table(title="Training envelope (warm-up excluded)")
    for c in ("Recording", "Valid", "Degraded", "Invalid", "Gap", "In envelope",
              "Leaves it most"):
        t.add_column(c)
    for r in rep["recordings"]:
        s = r["states"]
        top = ", ".join(f"{x['feature']} {x['pct']:g}%" for x in r["top_outside"][:3])
        pct = r["in_envelope_pct"]
        t.add_row(Path(r["recording"]).name, str(s["valid"]), str(s["degraded"]),
                  str(s["invalid"]), str(s["gap"]),
                  "—" if pct is None else (f"[red]{pct:g}%[/red]" if pct < 99 else f"{pct:g}%"),
                  top or "—")
    console.print(t)
    cols = envmod.by_column(rep["top_outside"], spec)
    if cols:
        console.print("Inputs (columns) leaving the envelope most: " +
                      ", ".join(f"{c} ({n} steps)" for c, n in cols.items()))
    if save:
        _write_or_exit(save, json.dumps(rep, indent=1))
        console.print(f"[green]Saved →[/green] {save}")


# ---------------------------------------------------------------- perturb

FAULTS_TEMPLATE = """\
# EdgeLens data faults: edgelens perturb spec.yaml data.csv --faults THIS_FILE
# Plain data: share it with whoever runs the same campaign on another tool.
seed: 1
faults:
  - name: imu-noise
    kind: noise          # noise | bias | gain | stuck | dropout | spike | gap | rate
    columns: ["angular_velocity.*", "linear_acceleration.*"]
    severity: [0.25, 0.5, 1, 2]     # noise sigma in training scales (IQR)
  - name: current-sensor-stuck
    kind: stuck
    columns: [current]
    severity: [1]
    start_s: 300
  - name: odometry-bias
    kind: bias
    columns: ["twist.twist.*"]
    severity: [0.5, 1, 2]
  - name: bus-dropout
    kind: dropout                   # rows lost, last value held
    columns: ["*"]
    severity: [0.1, 0.3, 0.5]
  - name: recorder-stall
    kind: gap
    severity: [5, 30, 120]          # seconds removed
    start_s: 400
  - name: half-rate
    kind: rate
    severity: [0.5, 0.75]           # rows per second vs training
"""


@app.command()
def perturb(spec_path: str = typer.Argument(..., help="Stream spec."),
            data: List[str] = typer.Argument(None, help="Recordings (CSV), with ground "
                                             "truth for detection metrics."),
            faults: str = typer.Option(None, "--faults", help="Fault file (YAML/JSON)."),
            init: str = typer.Option(None, "--init", help="Write a fault file template here "
                                     "and exit."),
            save: str = typer.Option(None, "--save", help="Robustness result (JSON)."),
            markdown: str = typer.Option(None, "--markdown", help="Robustness report (MD).")):
    """Inject data faults (noise, bias, stuck, dropout, spikes, gaps, rate changes)
    over a range of severities: robustness curves of accuracy, time to detect
    and false alarms, and the severity where the decision breaks."""
    if init:
        if Path(init).exists():
            console.print(f"[red]{init} exists; not overwriting.[/red]")
            raise typer.Exit(1)
        _write_or_exit(init, FAULTS_TEMPLATE)
        console.print(f"[green]Wrote[/green] {init}")
        return
    if not faults or not data:
        console.print("[red]Give recordings and --faults (or --init faults.yaml).[/red]")
        raise typer.Exit(2)
    spec = _spec_or_exit(spec_path)
    from .stream import perturb as pt
    try:
        fl = pt.load_faults(faults)
    except (OSError, ValueError) as e:
        console.print(f"[red]Faults: {e}[/red]")
        raise typer.Exit(2)
    recs = _recordings_or_exit(data, spec)
    model = _model_or_exit(spec)
    guard = _guard_or_exit(spec)
    try:
        doc = pt.sweep(recs, spec, model, fl, guard=guard,
                       progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    except pt.FaultError as e:
        console.print(f"[red]Faults: {e}[/red]")
        raise typer.Exit(2)
    for c in doc["curves"]:
        t = Table(title=f"{c['fault']['name']} — {c['meaning']}")
        for col in ("Severity", "RMSE", "Detect (max)", "Missed", "False alarms",
                    "In envelope (min)"):
            t.add_column(col)
        for p in c["points"]:
            t.add_row(f"{p['severity']:g}", str(p["rmse"]), _s(p["time_to_detect_max_s"]),
                      f"{p['missed']}/{p['events']}", str(p["false_alarms"]),
                      "—" if p["in_envelope_pct_min"] is None else f"{p['in_envelope_pct_min']:g}%")
        console.print(t)
        b = c["breaks_at"]
        console.print(f"  [red]breaks at severity {b['severity']:g}: {', '.join(b['why'])}[/red]"
                      if b else "  [green]no decision change over these severities[/green]")
    if save:
        _write_or_exit(save, json.dumps(doc, indent=1))
        console.print(f"[green]Saved →[/green] {save}")
    if markdown:
        _write_or_exit(markdown, pt.markdown(doc))
        console.print(f"[green]Report →[/green] {markdown}")


# ---------------------------------------------------------------- parity

@app.command()
def parity(spec_path: str = typer.Argument(..., help="Stream spec."),
           data: str = typer.Argument(..., help="One recording (CSV)."),
           reference: str = typer.Option(None, "--reference", help="Training-side model "
                                         "inputs for this recording: .npy, or float32 .bin "
                                         "[steps, sequence, width]."),
           reference_outputs: str = typer.Option(None, "--reference-outputs",
                                                 help="Another runtime's outputs (float32 "
                                                      ".bin / .npy), e.g. the C++ service."),
           steps: int = typer.Option(300, "--steps", help="Steps for the streaming self-check."),
           save: str = typer.Option(None, "--save")):
    """Train/serve parity: the deployed feature path vs the batch reference, vs
    the training code's own tensor, and the model outputs vs another runtime.
    Exit 0 = match, 1 = mismatch."""
    import numpy as np

    from .stream import parity as par
    from .stream.features import batch_features, windows
    spec = _spec_or_exit(spec_path)
    rec = _recordings_or_exit([data], spec)[0]
    rep = {"spec": spec["name"], "recording": data,
           "self_check": par.self_check(rec, spec, steps=steps)}
    bad = False
    t = Table(title="Parity")
    for c in ("Compared", "Against", "Steps", "Max |error|", "Result"):
        t.add_column(c)
    for mode, r in rep["self_check"].items():
        ok = r["max_abs_error"] < 1e-4
        bad |= not ok
        t.add_row(f"streaming ({mode})", "batch reference", str(r["steps"]),
                  f"{r['max_abs_error']:.2e}", "[green]match[/green]" if ok else "[red]MISMATCH[/red]")
    ours = None
    if reference or reference_outputs:
        F, _ = batch_features(rec, spec)
        ours = windows(F, spec).astype(np.float64)
    if reference:
        try:
            ref = par._load_tensor(reference, spec["features"]["sequence"],
                                   spec["features"]["width"])
        except (OSError, ValueError) as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(2)
        r = par.compare_tensor(ours, ref, spec)
        rep["reference"] = r
        ok = r["verdict"] == "match"
        bad |= not ok
        t.add_row("model inputs", Path(reference).name, f"{r['compared']}",
                  "—" if r.get("max_abs_error") is None else f"{r['max_abs_error']:.2e}",
                  "[green]match[/green]" if ok else "[red]MISMATCH[/red]")
    if reference_outputs:
        model = _model_or_exit(spec)
        raw = model.predict_many(ours.astype(np.float32))
        r = par.compare_outputs(raw, reference_outputs, spec)
        rep["outputs"] = r
        ok = (r["max_abs_error"] is not None and r["max_abs_error"] < 1e-3
              and not r.get("decisions_flipped") and r["steps_ours"] == r["steps_reference"])
        bad |= not ok
        t.add_row("model outputs", Path(reference_outputs).name,
                  f"{r['steps_ours']} / {r['steps_reference']}",
                  "—" if r["max_abs_error"] is None else f"{r['max_abs_error']:.2e}",
                  ("[green]match[/green]" if ok else "[red]MISMATCH[/red]")
                  + (f" ({r['decisions_flipped']} decisions flipped)"
                     if r.get("decisions_flipped") else ""))
    console.print(t)
    r = rep.get("reference") or {}
    for w in r.get("worst_features") or []:
        console.print(f"  worst: {w['feature']} |error| {w['max_abs_error']:.3g}")
    for s in r.get("signatures") or []:
        console.print(f"  [yellow]→ {s}[/yellow]")
    if save:
        _write_or_exit(save, json.dumps(rep, indent=1))
        console.print(f"[green]Saved →[/green] {save}")
    raise typer.Exit(1 if bad else 0)
