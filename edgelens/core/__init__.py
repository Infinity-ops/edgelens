"""EdgeLens core: the generic pipeline engine (scenario-independent)."""

from .observer import Tracer, trace
from .pipeline import ROLES, Pipeline, Stage, infer_role
from .schema import SCHEMA_VERSION

__all__ = ["Pipeline", "Stage", "ROLES", "infer_role", "Tracer", "trace", "SCHEMA_VERSION"]
