"""
edgelens.report.generator
----------------------------
Generates a single, self-contained HTML report (no external JS/CSS)
combining the hardware/software fingerprint, the stage-level latency
breakdown, and the diagnosis verdict. Designed to be attachable to a
GitHub issue or shared with a teammate as one file.

v0.1.0 changes from the alpha scaffold:
  - "confidence" -> "evidence strength" + the actual evidence numbers,
    matching edgelens.diagnose.engine's renamed output (see that module
    for why).
  - Demo-mode reports now carry a loud banner at BOTH the top and the
    bottom of the page, plus a repeating full-page watermark, so a
    synthetic report cannot be mistaken for — or accidentally passed
    along as — a real measurement.
"""

import datetime
import json
from pathlib import Path


def generate_html_report(fingerprint, benchmark, diagnosis, output_path):
    stages = benchmark.get("stage_avg_ms", {})
    total = benchmark.get("total_latency_ms", 0) or 1e-9
    mode = benchmark.get("mode", "unknown")
    is_demo = mode == "demo"

    bars = ""
    for stage, ms in stages.items():
        pct = (ms / total) * 100
        bars += f"""
        <div class="bar-row">
          <div class="bar-label">{stage.replace('_', ' ')}</div>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
          <div class="bar-value">{ms:.2f} ms &middot; {pct:.1f}%</div>
        </div>"""

    banner_html = ""
    watermark_html = ""
    if is_demo:
        warning_text = (
            "SIMULATED DATA \u2014 NOT A REAL MEASUREMENT. This report was generated "
            "with --demo. Every number on this page is synthetic, produced only to "
            "preview EdgeLens' output format. It reflects nothing about any real "
            "hardware or application. For a real measurement, run "
            "`edgelens benchmark --model your_model.onnx` on the target device."
        )
        banner_html = f'<div class="banner">\u26A0\uFE0F {warning_text}</div>'
        watermark_html = '<div class="watermark">SIMULATED &middot; NOT REAL &middot; SIMULATED &middot; NOT REAL</div>'

    secondary_html = ""
    for f in diagnosis.get("secondary", []):
        secondary_html += (
            f'<li><b>{f["type"]}</b> (evidence strength {f["evidence_strength"]*100:.0f}%) '
            f'\u2014 {f["detail"]}</li>'
        )

    raw_json = json.dumps(
        {"fingerprint": fingerprint, "benchmark": benchmark, "diagnosis": diagnosis},
        indent=2,
    )

    primary = diagnosis["primary"]
    evidence_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in (primary.get("evidence") or {}).items()
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{'[SIMULATED] ' if is_demo else ''}EdgeLens Report</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
          background:#0f1115; color:#e6e6e6; margin:0; padding:32px;
          position:relative; }}
  h1 {{ color:#5ee1a5; margin-bottom:4px; }}
  h2 {{ color:#cfd3da; font-size:16px; text-transform:uppercase;
        letter-spacing:0.04em; margin-bottom:14px; }}
  .card {{ background:#181b21; border:1px solid #2a2e37; border-radius:10px;
           padding:22px; margin-bottom:20px; position:relative; z-index:1; }}
  table {{ width:100%; border-collapse:collapse; }}
  td {{ padding:7px 10px; border-bottom:1px solid #2a2e37; font-size:14px; }}
  td:first-child {{ color:#8a8f98; width:220px; }}
  .bar-row {{ display:flex; align-items:center; margin-bottom:10px; font-size:14px; }}
  .bar-label {{ width:120px; text-transform:capitalize; color:#cfd3da; }}
  .bar-track {{ flex:1; background:#2a2e37; border-radius:4px; height:18px;
                overflow:hidden; margin:0 12px; }}
  .bar-fill {{ background:linear-gradient(90deg,#5ee1a5,#3fa1e8); height:100%; }}
  .bar-value {{ width:170px; font-size:13px; color:#9aa0aa; text-align:right; }}
  .verdict {{ font-size:22px; font-weight:700; color:#ffb454; margin-bottom:8px; }}
  .strength {{ color:#8a8f98; font-weight:400; font-size:15px; }}
  .rec {{ color:#5ee1a5; margin-top:10px; }}
  .meta {{ color:#666; font-size:13px; margin-bottom:24px; }}
  .banner {{ background:#4a1e1e; border:2px solid #ff5c5c; color:#ffb4b4;
             padding:14px 18px; border-radius:8px; margin-bottom:20px;
             font-size:14px; font-weight:600; position:relative; z-index:1; }}
  ul {{ padding-left:20px; font-size:14px; color:#cfd3da; }}
  pre {{ white-space:pre-wrap; font-size:11.5px; color:#8a8f98;
         max-height:420px; overflow:auto; }}
  .fpid {{ font-family: monospace; color:#5ee1a5; }}
  .evidence-table td {{ font-size:13px; }}
  .watermark {{
      position:fixed; top:0; left:0; width:100%; height:100%;
      display:flex; align-items:center; justify-content:center;
      transform:rotate(-25deg); font-size:64px; font-weight:800;
      color:rgba(255,92,92,0.08); pointer-events:none; z-index:0;
      white-space:nowrap; overflow:hidden;
  }}
</style>
</head>
<body>
  {watermark_html}
  <h1>{'[SIMULATED] ' if is_demo else ''}EdgeLens Report</h1>
  <div class="meta">Generated {datetime.datetime.now().isoformat(timespec='seconds')}
    &middot; board <span class="fpid">{fingerprint.get('hardware', {}).get('model', 'unknown')}</span></div>

  {banner_html}

  <div class="card">
    <h2>Hardware / Software Fingerprint</h2>
    <table>
      <tr><td>Board</td><td>{fingerprint['hardware']['model']}</td></tr>
      <tr><td>Is Jetson</td><td>{fingerprint['hardware']['is_jetson']}</td></tr>
      <tr><td>L4T / JetPack</td><td>{fingerprint['software']['jetpack_l4t'] or 'not detected'}</td></tr>
      <tr><td>CUDA</td><td>{fingerprint['software']['cuda'] or 'not detected'}</td></tr>
      <tr><td>TensorRT</td><td>{fingerprint['software']['tensorrt'] or 'not detected'}</td></tr>
      <tr><td>Python</td><td>{fingerprint['software']['python']}</td></tr>
      <tr><td>OS</td><td>{fingerprint['software']['os']}</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Pipeline Breakdown &middot; {mode} mode &middot; {benchmark.get('pipeline_source', 'unknown')}</h2>
    {bars}
    <p style="margin-top:16px;">Total latency: <b>{total:.2f} ms</b> &nbsp;|&nbsp;
       FPS: <b>{benchmark.get('fps', 0)}</b></p>
    <p class="meta">P50 {benchmark.get('latency_p50_ms')}ms &middot;
       P95 {benchmark.get('latency_p95_ms')}ms &middot;
       P99 {benchmark.get('latency_p99_ms')}ms &middot;
       {benchmark.get('iterations')} iterations</p>
  </div>

  <div class="card">
    <h2>Diagnosis</h2>
    <div class="verdict">{primary['type']} <span class="strength">(evidence strength {primary['evidence_strength']*100:.0f}%)</span></div>
    <p>{primary['detail']}</p>
    <p class="rec">&rarr; {primary['recommendation']}</p>
    {"<ul>" + secondary_html + "</ul>" if secondary_html else ""}
    <table class="evidence-table">
      {evidence_rows}
    </table>
  </div>

  <div class="card">
    <h2>Raw Data (JSON)</h2>
    <pre>{raw_json}</pre>
  </div>

  {banner_html}
</body>
</html>"""

    Path(output_path).write_text(html)
    return str(Path(output_path).resolve())
