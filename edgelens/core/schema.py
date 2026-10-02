"""Versioning for every JSON document EdgeLens writes.

Bump SCHEMA_VERSION on any breaking change to field names or meanings.
Readers treat a missing "schema_version" as 0 (pre-v0.1.0 files) and stay
backward compatible with them.
"""

SCHEMA_VERSION = 1


def version_of(doc):
    return (doc or {}).get("schema_version", 0)
