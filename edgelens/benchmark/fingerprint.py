"""
edgelens.benchmark.fingerprint
--------------------------------
Combines hardware + software + benchmark + diagnosis into one
reproducible JSON record ("fingerprint"). This is what lets two
developers or a bug report compare experiments apples-to-apples,
per the reproducibility principle in the project roadmap.
"""

import datetime
import hashlib
import json


def build_fingerprint(hw_sw_fingerprint, benchmark_result, diagnosis=None):
    record = {
        "edgelens_version": _version(),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "hardware": hw_sw_fingerprint["hardware"],
        "software": hw_sw_fingerprint["software"],
        "benchmark": benchmark_result,
        "diagnosis": diagnosis,
    }
    record["fingerprint_id"] = _hash_record(record)
    return record


def _version():
    from .. import __version__
    return __version__


def _hash_record(record):
    stable = json.dumps(
        {"hardware": record["hardware"], "software": record["software"]},
        sort_keys=True,
    )
    return hashlib.sha256(stable.encode()).hexdigest()[:12]
