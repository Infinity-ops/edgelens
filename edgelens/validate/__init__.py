"""Requirement validation: PASS / FAIL / INCONCLUSIVE with evidence."""

from .engine import REQUIREMENT_KEYS, exit_code, validate

__all__ = ["validate", "exit_code", "REQUIREMENT_KEYS"]
