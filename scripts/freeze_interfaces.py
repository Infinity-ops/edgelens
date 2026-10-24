"""Write tests/fixtures/frozen_v0_2.json: the interfaces frozen for 0.2.x."""
import json, sys
sys.path.insert(0, ".")
import typer
from edgelens.cli import app
from edgelens import contract as ct
from edgelens.validate import engine as ve
from edgelens.stream import spec as sp, perturb as pt
from edgelens.core import schema

def walk(c, prefix=""):
    out = {}
    if hasattr(c, "commands"):
        for name, sub in sorted(c.commands.items()):
            out.update(walk(sub, f"{prefix} {name}".strip()))
    else:
        out[prefix] = sorted(o for p in c.params for o in list(getattr(p, "opts", [])) + list(getattr(p, "secondary_opts", [])) if o.startswith("-"))
    return out

frozen = {
    "_comment": ("Interfaces frozen for EdgeLens 0.2.x (docs/stability.md). tests/test_frozen_interfaces.py "
                 "fails if any of these is removed or renamed; additions are allowed. Regenerate only "
                 "for a new minor version: python scripts/freeze_interfaces.py"),
    "cli": walk(typer.main.get_command(app)),
    "validate_requirements": sorted(ve.REQUIREMENT_KEYS),
    "contract_top_keys": sorted(ct.TOP_KEYS),
    "contract_assumptions": sorted(ct.ASSUMPTIONS),
    "stream_spec_top_keys": sorted(sp.TOP),
    "stream_spec_section_keys": {k: sorted(v) for k, v in sp.KEYS.items()},
    "perturb_fault_kinds": sorted(pt.KINDS),
    "result_schema_version": schema.SCHEMA_VERSION,
    "result_top_level_keys": sorted(["schema_version", "edgelens_version", "mode", "pipeline_source",
                                     "pipeline", "requirements", "iterations", "latency",
                                     "service_latency", "stage_stats_ms", "deadline", "backlog",
                                     "stage_avg_ms", "total_latency_ms", "fps", "throughput_basis",
                                     "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
                                     "telemetry", "telemetry_series", "energy", "trace", "clock",
                                     "identity", "environment", "protocol"]),
    "result_modes": sorted(schema.KNOWN_MODES),
}
json.dump(frozen, sys.stdout, indent=1, sort_keys=True)
print()
