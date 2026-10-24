"""The README quick start must keep working exactly as written: same
commands, same file, and the result passes check_result.py."""
import re
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
QS = ROOT / "examples" / "quickstart"


def _block(md, after):
    text = md[md.index(after):]
    return text[text.index("```bash") + 7:text.index("```", text.index("```bash") + 7)]


def test_readme_quick_start_matches_the_example():
    readme = (ROOT / "README.md").read_text()
    block = _block(readme, "## Quick start")
    assert "examples/quickstart/quickstart.py" in block
    example = (QS / "README.md").read_text()
    for line in block.splitlines():
        cmd = line.split("#")[0].strip()
        if cmd.startswith("edgelens "):
            assert cmd in example, f"README and examples/quickstart/README.md differ: {cmd}"


def test_quick_start_runs_and_passes_its_checks(tmp_path, monkeypatch):
    from edgelens.cli import app
    monkeypatch.chdir(tmp_path)
    (tmp_path / "quickstart.py").write_text((QS / "quickstart.py").read_text())
    r = CliRunner()
    res = r.invoke(app, ["benchmark", "--pipeline", "quickstart.py", "--iterations", "1000",
                         "--save", "run.json"])
    assert res.exit_code == 0, res.output
    assert r.invoke(app, ["diagnose", "run.json"]).exit_code == 0
    v = r.invoke(app, ["validate", "run.json", "--p99-ms", "32", "--max-miss-ratio", "0"])
    assert v.exit_code == 0, v.output
    chk = subprocess.run([sys.executable, str(QS / "check_result.py"), "run.json"],
                         capture_output=True, text=True)
    assert chk.returncode == 0 and chk.stdout.startswith("OK:"), chk.stdout + chk.stderr
    assert re.search(r"deadline met \(0 misses at 32.0 ms\)", chk.stdout)
