"""EdgeLens — measure, diagnose and validate AI pipelines on edge devices.

    import edgelens as el

    # harness mode: EdgeLens drives the loop
    pipe = el.Pipeline("bearing", pack="timeseries", sample_rate_hz=10_000, window=1024, hop=512)
    pipe.add_stage("filter", my_filter)
    pipe.add_stage("inference", my_model)
    result = pipe.run(iterations=1000, source=el.WindowSource("vib.npy", window=1024, hop=512))

    # observer mode: your loop, EdgeLens watches
    with el.trace("detector", pack="vision", deadline_ms=33) as t:
        with t.stage("inference"):
            ...
"""

__version__ = "0.1.0rc1"

from .core import SCHEMA_VERSION, Pipeline, Stage, Tracer, trace  # noqa: E402
from .packs import WindowSource, get_pack, list_packs, register_pack  # noqa: E402

__all__ = ["__version__", "Pipeline", "Stage", "Tracer", "trace", "SCHEMA_VERSION",
           "WindowSource", "get_pack", "list_packs", "register_pack"]
