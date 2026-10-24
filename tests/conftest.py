"""Shared test setup."""

import os
from pathlib import Path

import pytest

import edgelens

# Subprocess tests run `python app.py` / `python -m edgelens`. From a plain
# checkout (no `pip install -e .`) those children could not import edgelens
# and failed with ModuleNotFoundError; point them at the package under test.
_PKG_PARENT = str(Path(edgelens.__file__).resolve().parents[1])
os.environ["PYTHONPATH"] = os.pathsep.join(
    [_PKG_PARENT] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])


@pytest.fixture(autouse=True)
def _clean_kernel_log(monkeypatch):
    """Tests must not depend on the machine's real kernel log. Found on a Jetson
    Nano with a failing SD card: its 144 real storage errors changed the
    expected messages. A test that is ABOUT storage errors patches this itself."""
    from edgelens.hardware import storage
    monkeypatch.setattr(storage, "_kernel_log", lambda: ("[ 0.0] clean boot\n", "test"))


@pytest.fixture(autouse=True)
def _no_real_cards(monkeypatch, tmp_path):
    """No test reads this machine's SD-card IDs or writes the real cache."""
    from edgelens.hardware import storage
    monkeypatch.setattr(storage, "_card_ids", lambda: {})
    monkeypatch.setattr(storage, "_cache_path", lambda: str(tmp_path / "storage_errors.json"))
