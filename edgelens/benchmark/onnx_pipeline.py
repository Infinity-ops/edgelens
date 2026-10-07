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


def cuda_probe():
    """(usable, reason): can this process actually use the CUDA provider?

    get_available_providers() lists compiled-in providers only. Found on a
    real laptop: onnxruntime-gpu listed Tensorrt+CUDA, but libcublasLt was not
    installed; sessions silently fell back to CPU and the first device copy
    crashed. Allocating one CUDA tensor loads the provider library, which is
    the cheapest real check. Cached per process.
    """
    global _cuda_probe_result
    if _cuda_probe_result is None:
        if ort is None or "CUDAExecutionProvider" not in available_providers():
            _cuda_probe_result = (False, "CUDAExecutionProvider is not in this onnxruntime build")
        else:
            ort.set_default_logger_severity(4)      # our message replaces ORT's log spam
            try:
                ort.OrtValue.ortvalue_from_numpy(np.zeros(1, dtype=np.float32), "cuda", 0)
                _cuda_probe_result = (True, None)
            except Exception as e:
                msg = str(e)
                if "not available" in msg or "cannot open shared object" in msg:
                    # ORT's own text is a source path + C++ signature; say it plainly.
                    reason = "the CUDA/cuDNN libraries this onnxruntime-gpu build needs could not be loaded"
                else:
                    lines = msg.strip().splitlines()
                    reason = (lines[-1] if lines else type(e).__name__)[:200]
                _cuda_probe_result = (False, reason)
            finally:
                ort.set_default_logger_severity(3)
    return _cuda_probe_result


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


def _session_with(model_path, provider):
    """(session, fallback_message). onnxruntime does not fail when a provider
    can't be enabled: it silently falls back to CPU (and prints an 'EP Error'
    banner). Return the session plus what it printed, so the caller can
    check which provider is really active."""
    import contextlib
    import io
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        session = ort.InferenceSession(model_path, providers=[provider])
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

    def __init__(self, model_path, provider=None, input_shape=None, input_shapes=None):
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

        self.session = None
        for cand in candidates:
            try:
                session, banner = _session_with(model_path, cand)
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

    # ---- stage functions (each does real, timeable work) ----

    def capture(self):
        """Synthetic source: references pre-generated tensors (near-zero
        cost). Real capture latency can't be honestly approximated by a
        default — use --pipeline with your real source for that."""
        self._raw = self._static_inputs

    def preprocess(self):
        """Minimal, honest preprocessing: min-max normalise float inputs;
        integer/bool inputs pass through unchanged."""
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
        """Minimal, honest postprocessing: touch every output tensor."""
        for out in self._outputs:
            _ = np.asarray(out).sum()

    def stage_fns(self):
        return {
            "capture": self.capture,
            "preprocess": self.preprocess,
            "h2d_copy": self.h2d_copy,
            "inference": self.inference,
            "d2h_copy": self.d2h_copy,
            "postprocess": self.postprocess,
        }
