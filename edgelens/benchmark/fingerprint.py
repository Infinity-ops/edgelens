"""
edgelens.benchmark.fingerprint
--------------------------------
One reproducible JSON record combining environment, benchmark and
diagnosis — attachable to a bug report or a paper's artifact.

v0.1.0: carries the three identities (environment_id, experiment_id,
run_id). `fingerprint_id` is kept as an alias of environment_id for older
readers. The environment captured AT BENCHMARK TIME is preferred over the
machine generating the report — reports are often built elsewhere.
"""

import datetime

from ..core.identity import capture_environment, environment_id
from ..core.schema import SCHEMA_VERSION


def build_fingerprint(hw_sw_fingerprint, benchmark_result, diagnosis=None):
    env = benchmark_result.get("environment") or capture_environment(hw_sw_fingerprint)
    ident = dict(benchmark_result.get("identity") or {})
    ident.setdefault("environment_id", environment_id(env))
    record = {
        "schema_version": SCHEMA_VERSION,
        "edgelens_version": _version(),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "identity": ident,
        "environment": env,
        "hardware": hw_sw_fingerprint["hardware"],
        "software": hw_sw_fingerprint["software"],
        "benchmark": benchmark_result,
        "diagnosis": diagnosis,
    }
    record["fingerprint_id"] = ident["environment_id"]
    return record


def _version():
    from .. import __version__
    return __version__
