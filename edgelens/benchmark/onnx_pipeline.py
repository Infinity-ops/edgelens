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


def available_providers():
    if ort is None:
        return []
    return ort.get_available_providers()


def pick_provider(prefer=None):
    """Pick the best available ONNX Runtime execution provider.

    prefer: optional ordered list of provider names to try first.
    Returns None if onnxruntime has no providers available at all
    (should not happen — CPUExecutionProvider ships with every build).
    """
    avail = available_providers()
    if not avail:
        return None
    prefer = prefer or [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    for p in prefer:
        if p in avail:
            return p
    return avail[0]


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
        self.provider = provider or pick_provider()
        if self.provider is None:
            raise RuntimeError("No ONNX Runtime execution providers are available.")

        avail = available_providers()
        if self.provider not in avail:
            raise RuntimeError(
                f"Requested provider '{self.provider}' is not available. "
                f"Available providers: {avail}"
            )

        try:
            self.session = ort.InferenceSession(model_path, providers=[self.provider])
        except Exception as e:
            raise RuntimeError(f"onnxruntime could not load '{model_path}': {e}") from e

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
