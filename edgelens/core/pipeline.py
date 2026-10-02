"""
edgelens.core.pipeline
------------------------
The generic pipeline model: an ordered list of named stages. Nothing here
knows about cameras, sensors or LLMs — scenario packs (edgelens.packs) add
stage templates, metrics and diagnosis rules on top.

Each stage has a ROLE from a small fixed vocabulary, which is what the
generic diagnosis rules reason about:

    input        acquire data (camera grab, sensor read, file read)
    preprocess   transform before the model (resize, filter, FFT, features)
    transfer     host <-> device copies
    inference    the model itself
    postprocess  decode the model output (NMS, argmax, detokenize)
    decision     act on the result (threshold, publish, actuate)
    other        anything else

Roles are inferred from stage names when not given, and can always be set
explicitly: pipe.add_stage("fft", fft, role="preprocess").

Data flow: a stage function that takes one positional argument receives the
previous stage's return value (the first stage receives the item from the
run's `source`, if any). A zero-argument function is called with nothing —
the original EdgeLens stage_fns contract, which keeps working unchanged.
"""

import inspect
import re
from collections import OrderedDict

ROLES = ("input", "preprocess", "transfer", "inference", "postprocess", "decision", "other")

# Matched as PREFIXES of the name's tokens ("feature_extraction" ->
# ["feature", "extraction"]), so "act" can't fire inside "extraction".
# Roles are tried in this priority order: "denoise_model" is inference.
_ROLE_PREFIXES = (
    ("transfer", ("h2d", "d2h", "copy", "upload", "download", "transfer", "memcpy")),
    ("inference", ("infer", "model", "predict", "forward", "engine", "prefill",
                   "generate", "classif", "detect")),
    ("input", ("capture", "acquire", "acquisition", "grab", "read", "sensor", "camera",
               "ingest", "receive", "source")),
    ("postprocess", ("post", "nms", "argmax", "detoken", "track")),
    ("decision", ("decid", "decision", "act", "publish", "output", "alarm", "health",
                  "threshold", "rul", "control")),
    ("preprocess", ("pre", "resize", "normal", "filter", "fft", "stft", "feature",
                    "window", "decode", "letterbox", "tokeni", "resampl", "convert",
                    "denois", "extract", "scale", "crop", "mel", "spectr")),
)


def infer_role(name):
    tokens = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]
    for role, prefixes in _ROLE_PREFIXES:
        if any(t.startswith(p) for t in tokens for p in prefixes):
            return role
    return "other"


def _arity(fn):
    """Number of positional parameters fn wants (0 or 1 for our purposes)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return 0
    n = 0
    for p in sig.parameters.values():
        if p.kind == p.VAR_POSITIONAL:
            return 1
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty:
            n += 1
    return n


class Stage:
    __slots__ = ("name", "fn", "role", "device", "arity")

    def __init__(self, name, fn, role=None, device=None):
        if not isinstance(name, str) or not name:
            raise ValueError("Stage name must be a non-empty string.")
        if not callable(fn):
            raise TypeError(f"Stage '{name}' must be callable, got {type(fn).__name__}.")
        if role is not None and role not in ROLES:
            raise ValueError(f"Stage '{name}': unknown role '{role}'. Use one of {ROLES}.")
        self.name = name
        self.fn = fn
        self.role = role or infer_role(name)
        self.device = device
        self.arity = _arity(fn)
        if self.arity > 1:
            raise TypeError(
                f"Stage '{name}' takes {self.arity} required arguments; a stage takes "
                f"zero arguments, or one (the previous stage's output)."
            )

    def describe(self):
        d = {"name": self.name, "role": self.role}
        if self.device:
            d["device"] = self.device
        return d


class Pipeline:
    """An ordered, named sequence of stages.

        pipe = Pipeline("bearing-monitor", pack="timeseries",
                        sample_rate_hz=10_000, window=1024, hop=512)

        @pipe.stage
        def acquire(window): ...

        @pipe.stage(role="inference", device="gpu")
        def model(features): ...

        result = pipe.run(iterations=500, deadline_ms=20)
    """

    def __init__(self, name="pipeline", pack="custom", **config):
        from ..packs import get_pack  # local import: packs import core
        self.name = name
        self.pack = get_pack(pack)
        self.config = self.pack.configure(dict(config))
        self._stages = OrderedDict()
        self.source = None      # default input stream for run(); see set_source()

    # ---- building ----

    def add_stage(self, name, fn, role=None, device=None):
        if name in self._stages:
            raise ValueError(f"Duplicate stage name '{name}'.")
        self._stages[name] = Stage(name, fn, role=role, device=device)
        return self

    def stage(self, fn=None, *, name=None, role=None, device=None):
        """Decorator form: @pipe.stage or @pipe.stage(role=..., device=...)."""
        def register(f):
            self.add_stage(name or f.__name__, f, role=role, device=device)
            return f
        if fn is not None:
            return register(fn)
        return register

    @classmethod
    def from_stage_fns(cls, stage_fns, name="pipeline", pack=None, **config):
        """Build from the classic {name: zero-arg callable} dict. With no pack
        given, the six classic names select the vision pack, anything else
        the custom pack."""
        if not isinstance(stage_fns, dict) or not stage_fns:
            raise ValueError("stage_fns must be a non-empty dict of {name: callable}.")
        if pack is None:
            from ..packs.vision import VISION_STAGE_NAMES
            pack = "vision" if tuple(stage_fns) == VISION_STAGE_NAMES else "custom"
        pipe = cls(name=name, pack=pack, **config)
        for stage_name, fn in stage_fns.items():
            role = pipe.pack.role_for(stage_name)
            pipe.add_stage(stage_name, fn, role=role)
        return pipe

    def set_source(self, source):
        """Default input for run() when none is passed — lets a --pipeline
        script ship its own data (e.g. a WindowSource replaying a recording)."""
        self.source = source
        return self

    # ---- introspection ----

    @property
    def stages(self):
        return list(self._stages.values())

    @property
    def stage_names(self):
        return list(self._stages.keys())

    def __len__(self):
        return len(self._stages)

    def describe(self):
        return {
            "name": self.name,
            "pack": self.pack.name,
            "stages": [s.describe() for s in self.stages],
            "config": self.config,
        }

    # ---- running ----

    def run(self, iterations=100, warmup=10, source=None, deadline_ms=None,
            period_ms=None, pace=False, telemetry=True, telemetry_interval_s=0.1,
            idle_baseline_s=None, pipeline_source=None):
        """Harness mode: EdgeLens drives the loop. Returns a result dict
        (see edgelens.core.result.build_result)."""
        from .harness import run_harness
        return run_harness(
            self, iterations=iterations, warmup=warmup,
            source=source if source is not None else self.source,
            deadline_ms=deadline_ms, period_ms=period_ms, pace=pace,
            telemetry=telemetry, telemetry_interval_s=telemetry_interval_s,
            idle_baseline_s=idle_baseline_s, pipeline_source=pipeline_source,
            model_path=getattr(self, "_model_path", None),
        )
