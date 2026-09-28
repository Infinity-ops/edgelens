"""
edgelens.report.generator
----------------------------
Generates a single, self-contained HTML report (no external JS/CSS)
combining the hardware/software fingerprint, the stage-level latency
breakdown, and the diagnosis verdict. Designed to be attachable to a
GitHub issue or shared with a teammate as one file.
"""

import datetime
import json
from pathlib import Path


def generate_html_report(fingerprint, benchmark, diagnosis, output_path):
    stages = benchmark.get("stage_avg_ms", {})
    total = benchmark.get("total_latency_ms", 0) or 1e-9
    mode = benchmark.get("mode", "unknown")

    bars = ""
    for stage, ms in stages.items():
        pct = (ms / total) * 100
        bars += f"""
        <div class="bar-row">
          <div class="bar-label">{stage.replace('_', ' ')}</div>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
          <div class="bar-value">{ms:.2f} ms &middot; {pct:.1f}%</div>
        </div>"""

    demo_banner = ""
    if mode == "demo":
        demo_banner = (
            '<div class="banner">DEMO MODE — synthetic data for previewing '
            "EdgeLens output. Run on real Jetson hardware for actual measurements.</div>"
        )

    secondary_html = ""
    for f in diagnosis.get("secondary", []):
        secondary_html += f'<li><b>{f["type"]}</b> ({f["confidence"]*100:.0f}%) — {f["detail"]}</li>'

    raw_json = json.dumps(
        {"fingerprint": fingerprint, "benchmark": benchmark, "diagnosis": diagnosis},
        indent=2,
    )

    primary = diagnosis["primary"]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>EdgeLens Report</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
          background:#0f1115; color:#e6e6e6; margin:0; padding:32px; }}
  h1 {{ color:#5ee1a5; margin-bottom:4px; }}
  h2 {{ color:#cfd3da; font-size:16px; text-transform:uppercase;
        letter-spacing:0.04em; margin-bottom:14px; }}
  .card {{ background:#181b21; border:1px solid #2a2e37; border-radius:10px;
           padding:22px; margin-bottom:20px; }}
  table {{ width:100%; border-collapse:collapse; }}
  td {{ padding:7px 10px; border-bottom:1px solid #2a2e37; font-size:14px; }}
  td:first-child {{ color:#8a8f98; width:180px; }}
  .bar-row {{ display:flex; align-items:center; margin-bottom:10px; font-size:14px; }}
  .bar-label {{ width:120px; text-transform:capitalize; color:#cfd3da; }}
  .bar-track {{ flex:1; background:#2a2e37; border-radius:4px; height:18px;
                overflow:hidden; margin:0 12px; }}
  .bar-fill {{ background:linear-gradient(90deg,#5ee1a5,#3fa1e8); height:100%; }}
  .bar-value {{ width:170px; font-size:13px; color:#9aa0aa; text-align:right; }}
  .verdict {{ font-size:22px; font-weight:700; color:#ffb454; margin-bottom:8px; }}
  .conf {{ color:#8a8f98; font-weight:400; font-size:15px; }}
  .rec {{ color:#5ee1a5; margin-top:10px; }}
  .meta {{ color:#666; font-size:13px; margin-bottom:24px; }}
  .banner {{ background:#3a2e12; border:1px solid #6b5220; color:#ffcf6b;
             padding:10px 14px; border-radius:8px; margin-bottom:20px; font-size:13px; }}
  ul {{ padding-left:20px; font-size:14px; color:#cfd3da; }}
  pre {{ white-space:pre-wrap; font-size:11.5px; color:#8a8f98;
         max-height:420px; overflow:auto; }}
  .fpid {{ font-family: monospace; color:#5ee1a5; }}
</style>
</head>
<body>
  <h1>EdgeLens Report</h1>
  <div class="meta">Generated {datetime.datetime.now().isoformat(timespec='seconds')}
    &middot; fingerprint <span class="fpid">{fingerprint.get('hardware', {}).get('model', 'unknown')}</span></div>

  {demo_banner}

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
    <h2>Pipeline Breakdown &middot; {mode} mode</h2>
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
    <div class="verdict">{primary['type']} <span class="conf">({primary['confidence']*100:.0f}% confidence)</span></div>
    <p>{primary['detail']}</p>
    <p class="rec">&rarr; {primary['recommendation']}</p>
    {"<ul>" + secondary_html + "</ul>" if secondary_html else ""}
  </div>

  <div class="card">
    <h2>Raw Data (JSON)</h2>
    <pre>{raw_json}</pre>
  </div>
</body>
</html>"""

    Path(output_path).write_text(html)
    return str(Path(output_path).resolve())
