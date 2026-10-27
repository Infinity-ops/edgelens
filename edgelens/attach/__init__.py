"""Attach mode: observe applications EdgeLens doesn't run (no code changes)."""

from .procsampler import ProcessSampler, TargetError, find_process
from .runner import observe, to_result

__all__ = ["ProcessSampler", "TargetError", "find_process", "observe", "to_result"]
