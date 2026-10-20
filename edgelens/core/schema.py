"""
Versioning for every JSON document EdgeLens writes. See docs/schema.md.

Stability promise (frozen with v0.2): result documents are schema v1.
  * Fields are only ever ADDED within v1. A field's name, unit and meaning
    never change; readers must ignore fields they don't know.
  * Renaming, removing or redefining a field bumps SCHEMA_VERSION, and the
    readers in this package keep reading every older version.
  * Readers treat a missing "schema_version" as 0 (pre-v0.1.0 files).
  * tests/fixtures/result_v0_1_0.json — a file written by the published
    edgelens 0.1.0 — must keep loading in diagnose, validate, compare and
    report. That test is the promise.
"""

SCHEMA_VERSION = 1

# The fields a v1 result document always has (since 0.1.0). Everything else
# is optional and may be missing in older files.
REQUIRED_V1 = ("schema_version", "edgelens_version", "mode", "pipeline_source", "pipeline",
               "requirements", "iterations", "latency", "stage_stats_ms", "stage_avg_ms",
               "total_latency_ms", "fps", "telemetry", "energy", "trace")

# Added within v1, newest last: (field, first version that writes it).
ADDED_IN_V1 = (
    ("environment", "0.1.0"), ("identity", "0.1.0"), ("service_latency", "0.1.0"),
    ("clock", "0.2.0"), ("latency_series_ms", "0.2.0"), ("markers", "0.2.0"),
    ("process", "0.2.0"), ("process_series", "0.2.0"), ("observation", "0.2.0"),
    ("gc", "0.2.0"),
)

KNOWN_MODES = ("hardware", "demo", "markers", "attach", "imported-log", "ros2", "detect")


def version_of(doc):
    return (doc or {}).get("schema_version", 0)


def problems(doc):
    """Reasons this can't be read as a result document (empty list = fine).

    Deliberately lenient: readers need only `stage_avg_ms`, so hand-written
    or third-party documents keep working; use missing_v1_fields() to check
    that a writer produced a COMPLETE v1 document."""
    if not isinstance(doc, dict):
        return ["not a JSON object"]
    v = version_of(doc)
    if isinstance(v, int) and v > SCHEMA_VERSION:
        return [f"schema_version {v} is newer than this EdgeLens understands "
                f"({SCHEMA_VERSION}); upgrade edgelens"]
    if "stage_avg_ms" not in doc:
        return ["no 'stage_avg_ms': not an EdgeLens result"]
    return []


def missing_v1_fields(doc):
    """Required v1 fields a document lacks (what EdgeLens itself always writes)."""
    return [k for k in REQUIRED_V1 if k not in (doc or {})]
