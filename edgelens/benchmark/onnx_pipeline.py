"""
edgelens.benchmark.onnx_pipeline
-----------------------------------
Real inference pipeline stage functions backed by ONNX Runtime.

This replaces the v0.1.0-alpha placeholder (`time.sleep()` calls) with an
actual model execution: a real .onnx file is loaded, and each pipeline
stage (capture / preprocess / h2d_copy / inference / d2h_copy /
postprocess) does real work that is genuinely timed.

Execution provider selection:
    CPU laptop  -> CPUExecutionProvider (automatic)
    Jetson      -> CUDAExecutionProvider or TensorrtExecutionProvider,
                   if the JetPack onnxruntime-gpu wheel providing them
                   is installed; otherwise falls back to CPU.

Host<->device copy honesty:
    On a GPU execution provider, h2d_copy/inference/d2h_copy are measured
    as separate stages using ONNX Runtime's IOBinding API, so the copy
    cost is not hidden inside one big "inference" number.
    On a CPU-only provider, there IS no device to copy to/from — so
    h2d_copy/d2h_copy correctly report near-zero time. That is accurate
    behavior, not a bug: don't "fix" it to show a fake copy cost.

Swapping in your OWN model/camera/preprocessing:
    Build your own stage_fns dict (six callables: capture, preprocess,
    h2d_copy, inference, d2h_copy, postprocess) and pass it directly to
    edgelens.benchmark.runner.run_benchmark(stage_fns=...) instead of
    using --model. OnnxStagePipeline below is a ready-made, honest
    default for "I just have an .onnx file and want a real number,"
    not a requirement.
"""

import re

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - exercised via has_onnxruntime()
    ort = None
else:
    # Newer ONNX Runtime builds (1.20+) run a background device-discovery
    # probe at session-creation time that logs a WARNING when it can't
    # read a sysfs node like /sys/class/drm/cardN/device/vendor — common
    # on laptops/VMs with hybrid graphics, headless GPUs, or gaps in DRM
    # card numbering. It's benign (ORT still falls back correctly to
    # whichever execution provider was actually requested) but reads like
    # an EdgeLens error to anyone running this for the first time. Default
    # to ERROR-level logging so this internal probe stays quiet; nothing
    # here suppresses EdgeLens' own errors, which are raised as Python
    # exceptions, not ORT log lines.
    ort.set_default_logger_severity(3)


def has_onnxruntime() -> bool:
    return ort is not None


GPU_EXECUTION_PROVIDERS = ("TensorrtExecutionProvider", "CUDAExecutionProvider")
DEFAULT_PREFERENCE = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")

CUDA_UNLOADABLE_ADVICE = (
    "onnxruntime-gpu is installed, but its CUDA/cuDNN libraries are missing or the wrong "
    "version for this build, so the GPU providers cannot load. Install the CUDA and cuDNN "
    "versions your onnxruntime-gpu needs (onnxruntime.ai -> CUDA Execution Provider -> "
    "Requirements); on Jetson use the Jetson Zoo wheel matching your JetPack. For CPU-only "
    "use: pip uninstall onnxruntime-gpu && pip install onnxruntime"
)

_cuda_probe_result = None


def available_providers():
    """Providers this onnxruntime build was COMPILED with. Not proof that they
    can load: see usable_providers()."""
    if ort is None:
        return []
    return ort.get_available_providers()


# The probe runs in a CHILD process. Loading the CUDA provider can crash the
# whole interpreter (SIGBUS / SIGSEGV) when onnxruntime is broken — found on
# a real Jetson Nano where `pip install onnxruntime` had been laid over the
# Jetson onnxruntime-gpu wheel, mixing two builds in one package folder. A
# crash in the child is a diagnosis; in this process it would take every
# EdgeLens command down. Cached on disk per onnxruntime install.
# The probe does what a real benchmark does: create a CUDA session on a tiny
# model and run it once (v0.2.0.dev1 allocated a bare CUDA tensor instead, which
# crashes the Jetson Nano's onnxruntime-gpu 1.11 build although sessions work —
# a false "broken install"). Each STEP is printed first, so a crash names the
# step that crashed.
_PROBE_MODEL_B64 = ("CAQSDmVkZ2VsZW5zLXByb2JlOj8KDAoBeBIBeSIEUmVsdRIFcHJvYmVaEwoBeBIOCgwIARIICgIIAQoC"
                    "CARiEwoBeRIOCgwIARIICgIIAQoCCARCBAoAEAk=")      # Relu, x/y float[1,4]
_PROBE_CODE = r"""
import base64, sys
print("STEP import", flush=True)
import numpy as np, onnxruntime as ort
ort.set_default_logger_severity(4)
p = ort.get_available_providers()
print("LISTED " + ",".join(p), flush=True)
if "CUDAExecutionProvider" in p:
    print("STEP session", flush=True)
    s = ort.InferenceSession(base64.b64decode(sys.argv[1]),
                             providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    used = s.get_providers()
    print("USED " + ",".join(used), flush=True)
    if used and used[0] == "CUDAExecutionProvider":
        print("STEP run", flush=True)
        y = s.run(None, {"x": np.array([[-1.0, 2.0, -3.0, 4.0]], np.float32)})[0]
        if y.tolist() == [[0.0, 2.0, 0.0, 4.0]]:
            print("CUDA_OK", flush=True)
        else:
            print("CUDA_WRONG " + repr(y.tolist()), flush=True)
"""
PROBE_VERSION = 2               # part of the cache key: a new probe ignores old verdicts
PROBE_TIMEOUT_S = 180            # EDGELENS_PROBE_TIMEOUT_S overrides


def _probe_cache_path():
    import os
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "edgelens", "ort_cuda_probe.json")


_CUDA_DRIVER_GLOBS = ("/usr/lib/aarch64-linux-gnu/tegra/libcuda.so*",     # Jetson (JetPack 4/5)
                      "/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so*",    # Jetson (JetPack 6)
                      "/usr/lib/x86_64-linux-gnu/libcuda.so*",            # desktop
                      "/usr/lib64/libcuda.so*", "/usr/lib/wsl/lib/libcuda.so*")


def _cuda_driver_files():
    import glob
    import os
    out = {}
    for pattern in _CUDA_DRIVER_GLOBS:
        for f in glob.glob(pattern):
            real = os.path.realpath(f)
            try:
                st = os.stat(real)
            except OSError:
                continue
            out[real] = [st.st_size, int(st.st_mtime)]
    return out


def _boot_id():
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            return f.read().strip()
    except OSError:
        return None


def _install_key():
    """Identifies this onnxruntime install: versions of every onnxruntime
    distribution plus the mtime of its native library folder."""
    import os
    import sys
    key = {"probe": PROBE_VERSION, "python": sys.executable,
           "ort_version": getattr(ort, "__version__", None),
           "ort_file": getattr(ort, "__file__", None), "dists": ort_distributions(),
           # a driver reinstall / JetPack upgrade / reflash changes these ...
           "cuda_driver": _cuda_driver_files(),
           # ... and a reboot changes this: a verdict never outlives the boot
           "boot_id": _boot_id(),
           # the environment decides which CUDA libraries / devices load
           "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        capi = os.path.join(os.path.dirname(ort.__file__), "capi")
        key["capi_mtime"] = max(os.path.getmtime(os.path.join(capi, f))
                                for f in os.listdir(capi))
    except (OSError, ValueError, TypeError, AttributeError):
        key["capi_mtime"] = None
    return key


def _probe_timeout():
    import os
    try:
        return float(os.environ.get("EDGELENS_PROBE_TIMEOUT_S") or PROBE_TIMEOUT_S)
    except ValueError:
        return float(PROBE_TIMEOUT_S)


def _last_step(stdout):
    steps = [ln[5:].strip() for ln in (stdout or "").splitlines() if ln.startswith("STEP ")]
    return steps[-1] if steps else "start"


_STEP_TEXT = {"start": "starting Python", "import": "importing onnxruntime",
              "session": "creating a CUDA session", "run": "running a CUDA session"}


def _run_isolated_probe():
    import signal as _signal
    import subprocess
    import sys
    try:
        out = subprocess.run([sys.executable, "-c", _PROBE_CODE, _PROBE_MODEL_B64],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, timeout=_probe_timeout())
    except subprocess.TimeoutExpired as e:
        step = _last_step(e.stdout if isinstance(e.stdout, str) else
                          (e.stdout or b"").decode("utf-8", "replace"))
        return (False, f"{_STEP_TEXT.get(step, step)} hung for more than {_probe_timeout():g} s "
                       f"(test model, isolated process)", "hang")
    if out.returncode < 0:
        try:
            sig = _signal.Signals(-out.returncode).name
        except ValueError:
            sig = f"signal {-out.returncode}"
        step = _last_step(out.stdout)
        from ..hardware.storage import kernel_storage_errors
        st = kernel_storage_errors() if sig == "SIGBUS" else {"count": 0}
        if not st.get("count") and st.get("remembered"):
            r = st["remembered"][0]
            st = dict(st, count=r["count"], source=f"an earlier boot, {r['device']}",
                      lines=[r.get("last_line") or ""])
        if st.get("count"):
            return (False, f"{_STEP_TEXT.get(step, step)} crashed Python ({sig}), and the "
                           f"kernel log ({st['source']}) shows {st['count']} storage error(s), "
                           f"e.g. \"{st['lines'][-1][:120]}\". A failing SD card / disk makes "
                           f"memory-mapped libraries crash with SIGBUS: replace the card before "
                           f"reinstalling anything. EdgeLens uses the CPU provider meanwhile",
                    "crash")
        if len(ort_distributions()) > 1:
            why = ("two onnxruntime packages are installed over each other (see the fix "
                   "above)")
        else:
            why = ("the onnxruntime-gpu build does not match this JetPack / CUDA, or its "
                   "files were damaged by an earlier install; reinstall the wheel for your "
                   "JetPack: pip install --force-reinstall --no-deps <onnxruntime_gpu wheel>")
            if sig == "SIGBUS" and not st.get("readable", True):
                why += ("; SIGBUS is also what a failing SD card causes — check "
                        "`sudo dmesg | grep -i 'I/O error'` first")
        return (False, f"{_STEP_TEXT.get(step, step)} crashed Python ({sig}) in an isolated "
                       f"test process — {why}. EdgeLens itself is unaffected: it uses the CPU "
                       f"provider until this is fixed", "crash")
    if "CUDA_OK" in out.stdout:
        return (True, None, "ok")
    listed = next((ln[7:] for ln in out.stdout.splitlines() if ln.startswith("LISTED ")), None)
    if listed is not None and "CUDAExecutionProvider" not in listed.split(","):
        return (False, "CUDAExecutionProvider is not in this onnxruntime build", "absent")
    if "CUDA_WRONG" in out.stdout:
        return (False, "a CUDA session ran but returned wrong results (broken install)",
                "error")
    if "USED " in out.stdout and "CUDA_OK" not in out.stdout:
        used = next(ln[5:] for ln in out.stdout.splitlines() if ln.startswith("USED "))
        if "CUDAExecutionProvider" not in used.split(","):
            return (False, "the CUDA/cuDNN libraries this onnxruntime-gpu build needs could not "
                           "be loaded (a CUDA session fell back to the CPU)", "error")
    lines = (out.stderr or "").strip().splitlines()
    msg = lines[-1] if lines else "unknown error"
    if "not available" in msg or "cannot open shared object" in msg:
        msg = "the CUDA/cuDNN libraries this onnxruntime-gpu build needs could not be loaded"
    return (False, msg[:200], "error")


def forget_cuda_probe():
    """Drop the cached verdict (this process and on disk): the next call re-probes."""
    import os
    global _cuda_probe_result, _cuda_probe_meta
    _cuda_probe_result = None
    _cuda_probe_meta = {}
    try:
        os.remove(_probe_cache_path())
    except OSError:
        pass


_cuda_probe_meta = {}            # {"cached_at": iso time} when the verdict came from disk


def cuda_probe(force=False):
    """(usable, reason): can this machine actually use the CUDA provider?

    get_available_providers() lists compiled-in providers only (found on a
    real laptop: onnxruntime-gpu listed CUDA without libcublasLt installed).
    A CUDA session on a tiny model is created and run in a child process (see
    above).

    Only SUCCESS is cached on disk, keyed by the interpreter, the onnxruntime
    install, the CUDA driver file and the boot: a failure is exactly the result
    that changes once something is fixed, so it is re-tested every time (about
    1-2 s, and only on a system with a problem). force=True ignores the cache."""
    global _cuda_probe_result, _cuda_probe_meta
    if force:
        _cuda_probe_result = None
    if _cuda_probe_result is not None:
        return _cuda_probe_result[:2]
    _cuda_probe_meta = {}
    if ort is None or "CUDAExecutionProvider" not in available_providers():
        _cuda_probe_result = (False, "CUDAExecutionProvider is not in this onnxruntime build",
                              "absent")
        return _cuda_probe_result[:2]
    import datetime
    import json
    import os
    key = _install_key()
    path = _probe_cache_path()
    if not force:
        try:
            with open(path) as f:
                cached = json.load(f)
            if cached.get("key") == key and tuple(cached["result"])[2] == "ok":
                _cuda_probe_result = tuple(cached["result"])
                _cuda_probe_meta = {"cached_at": cached.get("saved_at")}
                return _cuda_probe_result[:2]
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            pass
    _cuda_probe_result = _run_isolated_probe()
    if _cuda_probe_result[2] == "ok":
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump({"key": key, "result": list(_cuda_probe_result),
                           "saved_at": datetime.datetime.now().isoformat(timespec="minutes")},
                          f)
        except OSError:
            pass
    else:
        try:
            os.remove(path)                  # a stale success must not come back
        except OSError:
            pass
    return _cuda_probe_result[:2]


def cuda_probe_cached_at():
    """When the current verdict was cached ("2026-10-17T16:12"), or None if it
    was measured in this process."""
    return _cuda_probe_meta.get("cached_at")


def cuda_probe_kind():
    """'ok' | 'absent' | 'error' | 'crash' | 'hang' (after cuda_probe())."""
    cuda_probe()
    return _cuda_probe_result[2]


def ort_distributions():
    """Installed distributions that provide the `onnxruntime` package."""
    try:
        from importlib import metadata
    except ImportError:                     # pragma: no cover
        return {}
    found = {}
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower()
        if name == "onnxruntime" or name.startswith("onnxruntime-") or name.startswith("onnxruntime_"):
            if name not in ("onnxruntime-extensions", "onnxruntime-tools"):
                found[name] = dist.version
    return found


def ort_install_problems():
    """Human-readable problems with the onnxruntime installation itself."""
    dists = ort_distributions()
    problems = []
    if len(dists) > 1:
        names = ", ".join(f"{k} {v}" for k, v in sorted(dists.items()))
        gpu = [k for k in dists if k != "onnxruntime"]
        keep = gpu[0] if gpu else "onnxruntime"
        problems.append({
            "problem": f"{len(dists)} onnxruntime packages are installed over each other "
                       f"({names}). They share one `onnxruntime` folder, so the files are a "
                       f"mix of two builds — this crashes or silently misbehaves.",
            "fix": f"pip uninstall -y {' '.join(sorted(dists))} && reinstall ONE of them "
                   f"(on Jetson: the {keep} wheel for your JetPack, e.g. from the Jetson "
                   f"Zoo). Never install edgelens with the [onnx] extra on a Jetson."})
    return problems


def unloadable_providers():
    """GPU providers that are listed but cannot load in this process."""
    listed = available_providers()
    gpu_listed = [p for p in listed if p in GPU_EXECUTION_PROVIDERS]
    if not gpu_listed or cuda_probe()[0]:
        return []
    return gpu_listed


def usable_providers():
    """Listed providers minus GPU providers whose libraries cannot load."""
    bad = set(unloadable_providers())
    return [p for p in available_providers() if p not in bad]


def pick_provider(prefer=None):
    """Best USABLE provider in preference order (TensorRT > CUDA > CPU).

    Returns None only if onnxruntime has no providers at all (should not
    happen — CPUExecutionProvider ships with every build).
    """
    usable = usable_providers()
    if not usable:
        return None
    for p in (prefer or DEFAULT_PREFERENCE):
        if p in usable:
            return p
    return usable[0]


def session_options(threads=None):
    """onnxruntime SessionOptions for a thread count, or None for its default
    (one intra-op thread per core). Found on an AGX Orin: a 2 ms LSTM ran as
    fast on 4 threads as on the default 8, at half the CPU and 18% less energy
    per inference; at one window per second 4 threads cut p99 by 32%."""
    if not threads:
        return None
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    return so


def _session_with(model_path, provider, options=None, sess_options=None):
    """(session, fallback_message). onnxruntime does not fail when a provider
    can't be enabled: it silently falls back to CPU (and prints an 'EP Error'
    banner). Return the session plus what it printed, so the caller can
    check which provider is really active. `options`: provider options, e.g.
    {"trt_fp16_enable": True} for TensorRT FP16."""
    import contextlib
    import io
    captured = io.StringIO()
    providers = [(provider, dict(options))] if options else [provider]
    with contextlib.redirect_stdout(captured):
        session = ort.InferenceSession(model_path, sess_options=sess_options,
                                       providers=providers)
    return session, captured.getvalue().strip()


# ONNX tensor type string -> NumPy dtype. Covers what edge models use.
_ORT_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def parse_input_shapes(specs):
    """Parse CLI --input-shape values into {name_or_None: (dims...)}.

    Accepted forms (repeatable):
        "1x8x2048"            -> applies to the model's only input
        "vib:1x8x2048"        -> applies to the input named "vib"
        "vib:1,8,2048"        -> commas work too
    """
    shapes = {}
    for spec in specs or []:
        spec = spec.strip()
        name, _, dims = spec.rpartition(":")
        try:
            parsed = tuple(int(d) for d in re.split(r"[x,]", dims) if d)
        except ValueError:
            raise RuntimeError(
                f"Invalid --input-shape '{spec}'. Expected e.g. 1x3x224x224 "
                f"or name:1x8x2048."
            )
        if not parsed or any(d <= 0 for d in parsed):
            raise RuntimeError(f"Invalid --input-shape '{spec}': dims must be positive integers.")
        shapes[name or None] = parsed
    return shapes


def _looks_like_image(shape):
    """NCHW with a fixed channel count of 1 or 3 -> safe to default H/W to 224."""
    return (len(shape) == 4 and isinstance(shape[1], int) and shape[1] in (1, 3))


def resolve_input_shape(name, declared, override=None):
    """Turn a declared (possibly symbolic) ONNX shape into concrete dims.

    Rules, in order:
      1. an explicit --input-shape override always wins (rank must match);
      2. a dynamic batch dim (index 0) defaults to 1;
      3. other dynamic dims default to 224 ONLY for image-like NCHW inputs;
      4. anything else raises: guessing a signal length or token count would
         silently benchmark a workload that doesn't exist.
    """
    if override is not None:
        if len(override) != len(declared):
            raise RuntimeError(
                f"--input-shape for '{name}' has rank {len(override)}, but the model "
                f"declares rank {len(declared)} {list(declared)}."
            )
        return tuple(override)

    resolved, unresolved = [], []
    for i, d in enumerate(declared):
        if isinstance(d, int) and d > 0:
            resolved.append(d)
        elif i == 0:
            resolved.append(1)
        elif _looks_like_image(declared):
            resolved.append(224)
        else:
            resolved.append(None)
            unresolved.append(i)
    if unresolved:
        example = "x".join(str(d) if d else "N" for d in resolved)
        raise RuntimeError(
            f"Input '{name}' has dynamic dimension(s) at index {unresolved} "
            f"(declared shape {list(declared)}) and EdgeLens will not guess them "
            f"for a non-image tensor.\nPass the real size, e.g.:\n"
            f"  --input-shape {name}:{example}"
        )
    return tuple(resolved)


def _synthetic_tensor(shape, dtype, rng):
    if np.issubdtype(dtype, np.floating):
        return rng.random(shape).astype(dtype)
    if dtype == np.bool_:
        return np.zeros(shape, dtype=np.bool_)
    # Small non-negative ints: safe for token ids / class indices / counts.
    return rng.integers(0, 10, size=shape).astype(dtype)


class OnnxStagePipeline:
    """Builds real stage functions around a real ONNX model for use with
    edgelens.benchmark.runner.run_benchmark(stage_fns=pipeline.stage_fns()).

    Every model input is fed (multi-input models are normal for sensor and
    sequence models), with the dtype the model declares.
    """

    GPU_PROVIDERS = ("CUDAExecutionProvider", "TensorrtExecutionProvider")

    def __init__(self, model_path, provider=None, input_shape=None, input_shapes=None,
                 provider_options=None, inputs=None, labels=None, threads=None,
                 inputs_order="sequential"):
        """inputs / labels: file paths (see edgelens.benchmark.real_inputs);
        threads: onnxruntime intra-op threads (None = its default)."""
        inputs_path = inputs          # `inputs` is reused below for the model's inputs
        if ort is None:
            raise RuntimeError(
                "onnxruntime is not installed. Install it with:\n"
                "  pip install onnxruntime        # CPU (any platform)\n"
                "  Jetson GPU: onnxruntime-gpu is NOT on PyPI for aarch64 — install\n"
                "  the wheel matching your JetPack + Python version from NVIDIA's\n"
                "  Jetson Zoo (elinux.org/Jetson_Zoo#ONNX_Runtime), after\n"
                "  `pip uninstall onnxruntime`.\n"
            )

        self.model_path = model_path
        # Providers tried and skipped in auto mode, with the reason; recorded
        # in the result so a CPU number is never mistaken for a GPU number.
        self.provider_fallbacks = []
        avail = available_providers()
        if provider is not None:
            # Explicit --provider: use exactly that one or fail loudly. A run
            # labelled CUDA that silently measured CPU is the worst outcome.
            if provider not in avail:
                raise RuntimeError(
                    f"Requested provider '{provider}' is not available. "
                    f"Available providers: {avail}"
                )
            if provider in unloadable_providers():
                raise RuntimeError(
                    f"'{provider}' is listed by onnxruntime but cannot load here "
                    f"({cuda_probe()[1]}).\n{CUDA_UNLOADABLE_ADVICE}\n"
                    f"Or measure on the CPU: --provider CPUExecutionProvider"
                )
            candidates, strict = [provider], True
        else:
            usable = usable_providers()
            if not usable:
                raise RuntimeError("No ONNX Runtime execution providers are available.")
            for p in unloadable_providers():
                self.provider_fallbacks.append(
                    {"provider": p, "reason": f"listed but cannot load: {cuda_probe()[1]}"})
            candidates = ([p for p in DEFAULT_PREFERENCE if p in usable]
                          + [p for p in usable if p not in DEFAULT_PREFERENCE])
            strict = False

        if threads is not None and int(threads) < 1:
            raise RuntimeError("--threads must be 1 or more (leave it out for onnxruntime's "
                               "default of one thread per core).")
        self.threads = int(threads) if threads else None
        sess_opts = session_options(self.threads)
        self.session = None
        for cand in candidates:
            try:
                opts = provider_options if (provider_options and cand == provider) else None
                extra = {}
                if opts:
                    extra["options"] = opts
                if sess_opts is not None:
                    extra["sess_options"] = sess_opts
                session, banner = _session_with(model_path, cand, **extra)
            except Exception as e:
                raise RuntimeError(f"onnxruntime could not load '{model_path}': {e}") from e
            active = session.get_providers()
            if cand in active:
                self.session, self.provider = session, cand
                break
            detail = next((ln.strip() for ln in banner.splitlines()
                           if ln.strip().startswith("EP Error")), "")[:300]
            if strict:
                raise RuntimeError(
                    f"onnxruntime could not enable '{cand}' and fell back to {active}, so this "
                    f"run would be labelled {cand} while measuring something else. "
                    f"{detail}\nInstall the libraries this provider needs (CUDA/cuDNN/"
                    f"TensorRT), or measure on the CPU: --provider CPUExecutionProvider"
                )
            self.provider_fallbacks.append(
                {"provider": cand, "reason": detail or f"fell back to {active}"})
        if self.session is None:
            raise RuntimeError("No ONNX Runtime execution provider could be enabled.")

        inputs = self.session.get_inputs()
        self.output_names = [o.name for o in self.session.get_outputs()]

        overrides = dict(input_shapes or {})
        if input_shape is not None:            # backward-compatible single-shape arg
            overrides[None] = tuple(input_shape)
        if None in overrides:
            if len(inputs) != 1:
                raise RuntimeError(
                    f"The model has {len(inputs)} inputs "
                    f"({', '.join(i.name for i in inputs)}); name each one: "
                    f"--input-shape NAME:DIMS."
                )
            overrides[inputs[0].name] = overrides.pop(None)
        unknown = set(overrides) - {i.name for i in inputs}
        if unknown:
            raise RuntimeError(
                f"--input-shape names {sorted(unknown)} are not model inputs. "
                f"Model inputs: {[i.name for i in inputs]}"
            )

        rng = np.random.default_rng(0)
        self.inputs = []      # [(name, shape, dtype)]
        self._static_inputs = {}
        for inp in inputs:
            dtype = _ORT_DTYPES.get(inp.type)
            if dtype is None:
                raise RuntimeError(
                    f"Input '{inp.name}' has unsupported type {inp.type}; use "
                    f"--pipeline with your own data for this model."
                )
            shape = resolve_input_shape(inp.name, inp.shape, overrides.get(inp.name))
            self.inputs.append((inp.name, shape, dtype))
            # Pre-generated ONCE: regenerating random data every iteration
            # costs ~1 ms on CNN-sized inputs and once looked like a fake
            # capture bottleneck (found on real hardware).
            self._static_inputs[inp.name] = _synthetic_tensor(shape, dtype, rng)

        from .real_inputs import InputError, Predictions, RealInputs, SampleOrder, load_labels
        from .real_inputs import _task as _label_task
        self.real = self.labels = self.preds = self.order = None
        self.labels_path = labels
        self._i = 0
        self._k = None
        try:
            if inputs_path:
                self.real = RealInputs(inputs_path, self.inputs)
                self.order = SampleOrder(self.real.n, inputs_order)
                self.preds = Predictions(self.real.n, self.session.get_outputs()[0].name)
                if labels:
                    self.labels = load_labels(labels, self.real.n)
                    oshape = self.session.get_outputs()[0].shape
                    if all(isinstance(d, int) and d > 0 for d in oshape):
                        _label_task(self.labels, int(np.prod(oshape)))   # fail before the run
            elif labels:
                raise InputError("--labels needs --inputs: labels belong to real input samples.")
        except InputError as e:
            raise RuntimeError(str(e)) from e

        # Backward-compatible attributes (first input).
        self.input_name = self.inputs[0][0]
        self.input_shape = self.inputs[0][1]

        self._is_gpu_provider = self.provider in self.GPU_PROVIDERS
        self._device = "cuda" if self._is_gpu_provider else "cpu"

        self._raw = None
        self._prepped = None
        self._device_inputs = None
        self._io_binding = None
        self._outputs = None

    def input_summary(self):
        return [{"name": n, "shape": list(s), "dtype": np.dtype(d).name} for n, s, d in self.inputs]

    def config(self):
        """What describes this run beyond the model: recorded in the result's
        pipeline config (and so in experiment_id) and compared by `compare`."""
        c = {"threads": self.threads or "default"}
        if self.real is not None:
            c["inputs"] = dict(self.real.describe(), order=self.order.order)
            if self.labels is not None:
                c["labels"] = self.labels_path
        else:
            c["inputs"] = "synthetic"
        return c

    def quality(self):
        """Accuracy against the labels over the samples predicted so far."""
        if self.labels is None or self.preds is None:
            return None
        return self.preds.quality(self.labels, self.labels_path)

    # ---- stage functions (each does real, timeable work) ----

    def capture(self):
        """Real inputs: the next sample, copied out of the file (what a
        receiver holds). Synthetic: references pre-generated tensors (near-zero
        cost); real capture latency can't be honestly approximated by a
        default — use --pipeline with your real source for that."""
        if self.real is not None:
            self._k = self.order.index(self._i)
            self._i += 1
            self._raw = self.real.sample(self._k)
        else:
            self._raw = self._static_inputs

    def preprocess(self):
        """Real inputs pass through unchanged (they are model inputs already,
        usually scaled). Synthetic: min-max normalise float inputs;
        integer/bool inputs pass through unchanged."""
        if self.real is not None:
            self._prepped = self._raw
            return
        out = {}
        for name, arr in self._raw.items():
            if np.issubdtype(arr.dtype, np.floating):
                span = np.ptp(arr)  # function form: ndarray.ptp() is gone in NumPy 2
                out[name] = ((arr - arr.min()) / (span + 1e-8)).astype(arr.dtype)
            else:
                out[name] = arr
        self._prepped = out

    def h2d_copy(self):
        if self._is_gpu_provider:
            self._device_inputs = {
                name: ort.OrtValue.ortvalue_from_numpy(arr, self._device, 0)
                for name, arr in self._prepped.items()
            }
        else:
            # CPU provider: no device to copy to. Correctly ~0 ms.
            self._device_inputs = self._prepped

    def inference(self):
        if self._is_gpu_provider:
            io = self.session.io_binding()
            for name, val in self._device_inputs.items():
                io.bind_ortvalue_input(name, val)
            for name in self.output_names:
                io.bind_output(name, self._device)
            self.session.run_with_iobinding(io)
            self._io_binding = io
        else:
            self._outputs = self.session.run(self.output_names, self._device_inputs)

    def d2h_copy(self):
        if self._is_gpu_provider:
            self._outputs = self._io_binding.copy_outputs_to_cpu()
        # CPU provider: outputs are already host-side from inference(). ~0 ms.

    def postprocess(self):
        """Minimal, honest postprocessing: touch every output tensor; with real
        inputs, keep this sample's prediction (preallocated, no allocation)."""
        for out in self._outputs:
            _ = np.asarray(out).sum()
        if self.preds is not None:
            self.preds.record(self._k, self._outputs)

    def stage_fns(self):
        return {
            "capture": self.capture,
            "preprocess": self.preprocess,
            "h2d_copy": self.h2d_copy,
            "inference": self.inference,
            "d2h_copy": self.d2h_copy,
            "postprocess": self.postprocess,
        }
