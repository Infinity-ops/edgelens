"""Scenario packs. See edgelens.packs.base.Pack for what a pack is."""

from .base import CustomPack, Pack
from .timeseries import TimeseriesPack, WindowSource
from .vision import VISION_STAGE_NAMES, VisionPack

_PACKS = {p.name: p for p in (CustomPack(), VisionPack(), TimeseriesPack())}


def get_pack(name):
    if isinstance(name, Pack):
        return name
    key = (name or "custom").lower()
    if key not in _PACKS:
        raise ValueError(f"Unknown pack '{name}'. Available: {', '.join(sorted(_PACKS))}")
    return _PACKS[key]


def list_packs():
    return list(_PACKS.values())


def register_pack(pack):
    """Register a third-party pack instance (must subclass Pack)."""
    if not isinstance(pack, Pack):
        raise TypeError("register_pack() expects a Pack instance")
    _PACKS[pack.name.lower()] = pack


__all__ = ["Pack", "get_pack", "list_packs", "register_pack", "WindowSource",
           "VISION_STAGE_NAMES", "VisionPack", "TimeseriesPack", "CustomPack"]
