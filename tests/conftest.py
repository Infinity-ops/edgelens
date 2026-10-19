"""Shared test setup."""

import pytest


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
