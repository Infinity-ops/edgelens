# Stability: what you can rely on in 0.2.x

EdgeLens 0.2 is an **Alpha**: the measurements are tested and parts are
verified on hardware ([jetson.md](jetson.md)), but the tool is young. To make
it safe to build CI gates, contracts and scripts on it, these interfaces are
**frozen for every 0.2.x release**: things may be added, nothing is removed
or renamed, and existing meanings don't change. `tests/test_frozen_interfaces.py`
checks this against a snapshot (`tests/fixtures/frozen_v0_2.json`) on every
commit.

| Interface | Frozen in 0.2.x |
| --- | --- |
| Command line | every command and every option listed by `edgelens --help` and `edgelens <command> --help` in 0.2.0 |
| Exit codes | `validate` and `contract run`: 0 PASS, 1 FAIL, 2 INCONCLUSIVE; `compare`: 0 pass, 1 regression, 2 environments differ (`--strict-env`) |
| Result files | `schema_version: 1` ([schema.md](schema.md)). Fields are only added; a file written by 0.1.0 or 0.2.x loads in every 0.2.x command. A reader must ignore fields it does not know |
| Requirements | the `validate` flags and contract guarantee keys (`p99_ms`, `max_miss_ratio`, `max_energy_mj`, `max_rmse`, …) and what they measure |
| Contracts | the file format: top-level keys, assumption keys, `{limit, id, note}` / `{value, id}` forms ([contracts.md](contracts.md)) |
| Stream specs and fault files | the YAML keys of `stream init` specs and `perturb` fault files ([stream.md](stream.md)) |
| Python API | `edgelens.Pipeline`, `Pipeline.stage` / `add_stage` / `run`, `edgelens.trace`, `edgelens.mark`, `edgelens.WindowSource` |
| C/C++ markers | `edgelens_mark.h`: `el_begin` / `el_end`, `EL_SPAN`, `EL_ITERATION`, `EL_SPAN_GPU`, `EL_SPAN_CUDA`, `-DEDGELENS_DISABLE`; the datagram protocol between senders and the collector |

**Experimental** (may change in a 0.2.x release; their file formats above
stay frozen): the reports and metrics of `stream`, `replay`, `detect`,
`envelope`, `perturb`, `parity` and `ros2 latency`, and every `diagnose`
finding's wording and score. Experimental commands say so in their `--help`.

**Not an interface:** console layout and wording (parse the JSON, not the
table), the HTML report, `examples/`, `scripts/` and anything under
`edgelens._*` or not listed above.

A change that would break a frozen interface waits for 0.3 and is listed
under "Upgrading" in the changelog. Bug fixes that make a number *correct*
are not breaking changes, even when the number changes; they are listed under
"Fixed" with the old and the new behaviour.
