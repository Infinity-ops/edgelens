"""The 0.2.x interface freeze (docs/stability.md): commands, options, contract
and stream-spec keys and the result's top-level fields may be ADDED in a
0.2.x release, never removed or renamed. A failure here means a change would
break someone's scripts, contracts, CI gates or result readers: keep the old
name (as an alias) or wait for 0.3."""
import json
from pathlib import Path

import typer

FROZEN = json.loads((Path(__file__).parent / "fixtures" / "frozen_v0_2.json").read_text())


def _cli():
    from edgelens.cli import app

    def walk(c, prefix=""):
        out = {}
        if hasattr(c, "commands"):
            for name, sub in sorted(c.commands.items()):
                out.update(walk(sub, f"{prefix} {name}".strip()))
        else:
            out[prefix] = {o for p in c.params for o in list(getattr(p, "opts", []))
                           + list(getattr(p, "secondary_opts", [])) if o.startswith("-")}
        return out
    return walk(typer.main.get_command(app))


def test_no_command_or_option_was_removed():
    now = _cli()
    gone = [c for c in FROZEN["cli"] if c not in now]
    assert not gone, f"commands removed: {gone}"
    lost = {c: sorted(set(opts) - now[c]) for c, opts in FROZEN["cli"].items()
            if set(opts) - now[c]}
    assert not lost, f"options removed: {lost}"


def test_requirement_contract_and_stream_keys_still_accepted():
    from edgelens import contract as ct
    from edgelens.stream import perturb as pt
    from edgelens.stream import spec as sp
    from edgelens.validate import engine as ve
    assert set(FROZEN["validate_requirements"]) <= set(ve.REQUIREMENT_KEYS)
    assert set(FROZEN["contract_top_keys"]) <= set(ct.TOP_KEYS)
    assert set(FROZEN["contract_assumptions"]) <= set(ct.ASSUMPTIONS)
    assert set(FROZEN["stream_spec_top_keys"]) <= set(sp.TOP)
    for section, keys in FROZEN["stream_spec_section_keys"].items():
        assert set(keys) <= set(sp.KEYS.get(section, ())), section
    assert set(FROZEN["perturb_fault_kinds"]) <= set(pt.KINDS)


def test_result_schema_version_and_fields():
    from edgelens.benchmark.runner import run_benchmark
    from edgelens.core import schema
    assert schema.SCHEMA_VERSION == FROZEN["result_schema_version"]
    assert set(FROZEN["result_modes"]) <= set(schema.KNOWN_MODES)
    r = run_benchmark(iterations=3, warmup=0, stage_fns={"a": lambda: None})
    missing = set(FROZEN["result_top_level_keys"]) - set(r)
    assert not missing, f"result fields removed: {sorted(missing)}"
