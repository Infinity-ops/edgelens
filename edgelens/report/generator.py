"""
edgelens.report.generator
----------------------------
A single, self-contained HTML report (no external JS/CSS): environment and
identities, latency/tail/deadline, energy, the stage breakdown, scenario
metrics and the diagnosis with its evidence. Attachable to a GitHub issue.

Every value that came from data (stage names, paths, model names, findings)
is HTML-escaped: a stage or file name containing "<" must never become
markup.

Demo reports carry a loud banner at the top and bottom plus a full-page
watermark, so a synthetic report can't be mistaken for a real measurement.
"""

import datetime
import json
from html import escape
from pathlib import Path


def _e(value):
    return escape("" if value is None else str(value))


def _fmt(v, unit=""):
    if v is None:
        return '<span class="dim">n/a</span>'
    if isinstance(v, float):
        return _e(f"{v:.3f}".rstrip("0").rstrip(".") + unit)
    return _e(f"{v}{unit}")


def _rows(pairs):
    return "".join(f"<tr><td>{_e(k)}</td><td>{v}</td></tr>" for k, v in pairs)


def generate_html_report(fingerprint, benchmark, diagnosis, output_path):
    stages = benchmark.get("stage_avg_ms", {}) or {}
    stage_sum = sum(stages.values()) or 1e-9
    mode = benchmark.get("mode", "unknown")
    is_demo = mode == "demo"
    pipe = benchmark.get("pipeline") or {}
    roles = {s["name"]: s.get("role", "") for s in pipe.get("stages", [])}
    stats = benchmark.get("stage_stats_ms") or {}

    bars = ""
    for stage, ms in stages.items():
        pct = (ms / stage_sum) * 100
        p99 = (stats.get(stage) or {}).get("p99")
        bars += f"""
        <div class="bar-row">
          <div class="bar-label">{_e(stage)} <span class="role">{_e(roles.get(stage, ''))}</span></div>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
          <div class="bar-value">{ms:.3f} ms &middot; {pct:.1f}%{' &middot; p99 ' + _e(p99) if p99 is not None else ''}</div>
        </div>"""

    lat = benchmark.get("latency") or {}
    lat_rows = _rows([(k, _fmt(lat.get(k), " ms")) for k in
                      ("min", "mean", "p50", "p90", "p95", "p99", "p99_9", "max", "jitter",
                       "tail_spread")])
    if lat.get("insufficient_samples"):
        lat_rows += _rows([("not reported", _e("; ".join(lat["insufficient_samples"])))])

    dl = benchmark.get("deadline")
    if dl:
        status = ('<span class="ok">MET</span>' if dl["met"]
                  else '<span class="bad">MISSED</span>')
        deadline_html = f"""<div class="card"><h2>Deadline</h2><table>{_rows([
            ("status", status),
            ("deadline", _fmt(dl["deadline_ms"], " ms")),
            ("misses", _e(f"{dl['misses']} / {dl['iterations']} ({dl['miss_ratio']*100:.3f}%)")),
            ("worst", _fmt(dl["worst_ms"], " ms")),
            ("worst overrun", _fmt(dl["worst_overrun_ms"], " ms")),
            ("longest miss burst", _fmt(dl["max_consecutive_misses"])),
            ("median slack", _fmt(dl.get("slack_p50_ms"), " ms")),
        ])}</table></div>"""
    else:
        deadline_html = ""

    en = benchmark.get("energy") or {}
    if en.get("available"):
        energy_html = f"""<div class="card"><h2>Power &amp; energy</h2><table>{_rows([
            ("mean power", _fmt(en.get("power_w_mean"), " W")),
            ("energy / iteration", _fmt(en.get("energy_per_iteration_j"), " J")),
            ("iterations / joule", _fmt(en.get("iterations_per_joule"))),
            ("idle power", _fmt(en.get("idle_power_w"), " W")),
            ("dynamic energy / iteration", _fmt(en.get("dynamic_energy_per_iteration_j"), " J")),
            ("method", _e(en.get("method"))),
        ])}</table>{('<p class="warn">' + _e(en["low_confidence"]) + '</p>') if en.get("low_confidence") else ''}
        <p class="dim">On-module INA3221 sensor readings, not a calibrated power meter.</p></div>"""
    else:
        energy_html = ""

    pm = benchmark.get("pack_metrics") or {}
    pack_html = ""
    if pm:
        pack_html = (f'<div class="card"><h2>{_e(pipe.get("pack", ""))} pack metrics</h2><table>'
                     + _rows([(k, _e(json.dumps(v) if isinstance(v, dict) else v))
                              for k, v in pm.items()]) + "</table></div>")

    env = fingerprint.get("environment") or {}
    ident = fingerprint.get("identity") or {}
    pmode = env.get("power_mode") or {}
    env_rows = _rows([
        ("Board", _e(env.get("board"))),
        ("L4T / JetPack", _e(env.get("l4t") or "not detected")),
        ("CUDA", _e(env.get("cuda") or "not detected")),
        ("TensorRT", _e(env.get("tensorrt") or "not detected")),
        ("Power mode", _e(pmode.get("name") or pmode.get("id") or "not detected")),
        ("Clocks locked (CPU / GPU)", _e(f"{env.get('cpu_clocks_locked')} / {env.get('gpu_clocks_locked')}")),
        ("Python", _e(env.get("python"))),
        ("OS", _e(env.get("os"))),
        ("environment_id", f'<span class="fpid">{_e(ident.get("environment_id"))}</span>'),
        ("experiment_id", f'<span class="fpid">{_e(ident.get("experiment_id"))}</span>'),
        ("run_id", f'<span class="fpid">{_e(ident.get("run_id"))}</span>'),
    ])

    banner_html = watermark_html = ""
    if is_demo:
        warning_text = (
            "SIMULATED DATA \u2014 NOT A REAL MEASUREMENT. This report was generated "
            "with --demo. Every number on this page is synthetic, produced only to "
            "preview EdgeLens' output format. It reflects nothing about any real "
            "hardware or application. For a real measurement, run "
            "`edgelens benchmark --model your_model.onnx` on the target device."
        )
        banner_html = f'<div class="banner">\u26A0\uFE0F {_e(warning_text)}</div>'
        watermark_html = ('<div class="watermark">SIMULATED &middot; NOT REAL &middot; '
                          'SIMULATED &middot; NOT REAL</div>')

    primary = diagnosis["primary"]
    secondary_html = "".join(
        f'<li><b>{_e(f["type"])}</b> ({_e(f["evidence_strength"])} evidence) '
        f'\u2014 {_e(f["detail"])}</li>' for f in diagnosis.get("secondary", []))
    evidence_rows = _rows([(k, _e(json.dumps(v) if isinstance(v, (dict, list)) else v))
                           for k, v in (primary.get("evidence") or {}).items()])

    raw_json = _e(json.dumps({"fingerprint": {k: v for k, v in fingerprint.items()
                                              if k != "benchmark"},
                              "diagnosis": diagnosis,
                              "benchmark": {k: v for k, v in benchmark.items()
                                            if k not in ("trace", "telemetry_series")}},
                             indent=2, default=str))
    trace_note = _e(f"{(benchmark.get('trace') or {}).get('event_count', 0)} trace events and "
                    f"{len(benchmark.get('telemetry_series') or [])} telemetry samples are in "
                    f"the benchmark JSON (omitted here).")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{'[SIMULATED] ' if is_demo else ''}EdgeLens Report</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
          background:#0f1115; color:#e6e6e6; margin:0; padding:32px; position:relative; }}
  h1 {{ color:#5ee1a5; margin-bottom:4px; }}
  h2 {{ color:#cfd3da; font-size:16px; text-transform:uppercase;
        letter-spacing:0.04em; margin-bottom:14px; }}
  .card {{ background:#181b21; border:1px solid #2a2e37; border-radius:10px;
           padding:22px; margin-bottom:20px; position:relative; z-index:1; }}
  table {{ width:100%; border-collapse:collapse; }}
  td {{ padding:7px 10px; border-bottom:1px solid #2a2e37; font-size:14px; }}
  td:first-child {{ color:#8a8f98; width:240px; }}
  .bar-row {{ display:flex; align-items:center; margin-bottom:10px; font-size:14px; }}
  .bar-label {{ width:200px; color:#cfd3da; }}
  .role {{ color:#6b7280; font-size:12px; }}
  .bar-track {{ flex:1; background:#2a2e37; border-radius:4px; height:18px;
                overflow:hidden; margin:0 12px; }}
  .bar-fill {{ background:linear-gradient(90deg,#5ee1a5,#3fa1e8); height:100%; }}
  .bar-value {{ width:240px; font-size:13px; color:#9aa0aa; text-align:right; }}
  .verdict {{ font-size:22px; font-weight:700; color:#ffb454; margin-bottom:8px; }}
  .strength {{ color:#8a8f98; font-weight:400; font-size:15px; }}
  .rec {{ color:#5ee1a5; margin-top:10px; }}
  .meta, .dim {{ color:#777; font-size:13px; }}
  .ok {{ color:#5ee1a5; font-weight:700; }} .bad {{ color:#ff5c5c; font-weight:700; }}
  .warn {{ color:#ffb454; font-size:13px; }}
  .banner {{ background:#4a1e1e; border:2px solid #ff5c5c; color:#ffb4b4;
             padding:14px 18px; border-radius:8px; margin-bottom:20px;
             font-size:14px; font-weight:600; position:relative; z-index:1; }}
  ul {{ padding-left:20px; font-size:14px; color:#cfd3da; }}
  pre {{ white-space:pre-wrap; font-size:11.5px; color:#8a8f98; max-height:420px; overflow:auto; }}
  .fpid {{ font-family: monospace; color:#5ee1a5; }}
  .watermark {{ position:fixed; top:0; left:0; width:100%; height:100%; display:flex;
      align-items:center; justify-content:center; transform:rotate(-25deg); font-size:64px;
      font-weight:800; color:rgba(255,92,92,0.08); pointer-events:none; z-index:0;
      white-space:nowrap; overflow:hidden; }}
</style>
</head>
<body>
  {watermark_html}
  <h1>{'[SIMULATED] ' if is_demo else ''}EdgeLens Report</h1>
  <div class="meta">Generated {datetime.datetime.now().isoformat(timespec='seconds')}
    &middot; {_e(env.get('board'))} &middot; pipeline <b>{_e(pipe.get('name'))}</b>
    ({_e(pipe.get('pack'))} pack)</div>
  {banner_html}

  <div class="card"><h2>Environment &amp; identity</h2><table>{env_rows}</table></div>

  <div class="card">
    <h2>Pipeline breakdown &middot; {_e(mode)} mode &middot; {_e(benchmark.get('pipeline_source', 'unknown'))}</h2>
    {bars}
    <p style="margin-top:16px;">Mean end-to-end: <b>{_fmt(benchmark.get('total_latency_ms'), ' ms')}</b>
       &nbsp;|&nbsp; Throughput: <b>{_fmt(benchmark.get('fps'))}</b> iterations/s
       &nbsp;|&nbsp; {_e(benchmark.get('iterations'))} iterations</p>
  </div>

  <div class="card"><h2>End-to-end latency</h2><table>{lat_rows}</table></div>
  {deadline_html}
  {energy_html}
  {pack_html}

  <div class="card">
    <h2>Diagnosis</h2>
    <div class="verdict">{_e(primary['type'])} <span class="strength">({_e(primary['evidence_strength'])} evidence)</span></div>
    <p>{_e(primary['detail'])}</p>
    <p class="rec">&rarr; {_e(primary['recommendation'])}</p>
    {"<ul>" + secondary_html + "</ul>" if secondary_html else ""}
    <table>{evidence_rows}</table>
  </div>

  <div class="card"><h2>Raw data (JSON)</h2><p class="dim">{trace_note}</p><pre>{raw_json}</pre></div>
  {banner_html}
</body>
</html>"""

    # utf-8 explicitly: the page declares <meta charset="utf-8"> and contains
    # non-ASCII (—, →, ⚠); the locale default crashed under LANG=C.
    Path(output_path).write_text(html, encoding="utf-8")
    return str(Path(output_path).resolve())
