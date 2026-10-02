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


class OnnxStagePipeline:
    """Builds real stage functions around a real ONNX model for use with
    edgelens.benchmark.runner.run_benchmark(stage_fns=pipeline.stage_fns()).
    """

    GPU_PROVIDERS = ("CUDAExecutionProvider", "TensorrtExecutionProvider")

    def __init__(self, model_path, provider=None, input_shape=None):
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

        self.session = ort.InferenceSession(model_path, providers=[self.provider])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        shape = input_shape or self.session.get_inputs()[0].shape
        resolved = []
        for i, d in enumerate(shape):
            if isinstance(d, int) and d > 0:
                resolved.append(d)
            else:
                # dynamic/symbolic dim (batch size, or unspecified) -> pick a
                # sane default so the benchmark can actually run
                resolved.append(1 if i == 0 else 224)
        self.input_shape = tuple(resolved)

        self._is_gpu_provider = self.provider in self.GPU_PROVIDERS
        self._device = "cuda" if self._is_gpu_provider else "cpu"

        # Pre-generate ONE random frame at construction time. The
        # capture() stage below just references it (near-zero cost) rather
        # than calling np.random.rand() fresh every iteration.
        # WHY THIS MATTERS: np.random.rand() on a realistic CNN input size
        # (e.g. 1x3x224x224) costs roughly 1ms per call on CPU — enough to
        # rival or exceed a small model's actual inference time. Generating
        # a "fresh random frame" every iteration was making the synthetic
        # capture() stage look like a real bottleneck when it was actually
        # just measuring NumPy's RNG cost, not anything resembling a real
        # camera/video capture. Found via real hardware testing against a
        # more realistically-sized model than the tiny test fixture.
        self._static_frame = np.random.rand(*self.input_shape).astype(np.float32)

        self._raw_frame = None
        self._prepped = None
        self._device_input = None
        self._io_binding = None
        self._outputs = None

    # ---- stage functions (each does real, timeable work) ----

    def capture(self):
        """Synthetic frame source: references a pre-generated frame rather
        than regenerating random data every call (see __init__ for why).
        Swap for a real camera/video capture by building your own
        stage_fns dict instead of using --model — this stage exists so
        the pipeline has SOME input ready, not to simulate real capture
        latency, which varies enormously by camera/decoder and can't be
        honestly approximated by a synthetic default."""
        self._raw_frame = self._static_frame

    def preprocess(self):
        """Minimal, honest preprocessing: min-max normalize to [0,1].
        Real pipelines (resize, color convert, letterbox, etc.) should
        replace this via a custom stage_fns dict."""
        frame = self._raw_frame
        span = np.ptp(frame)  # np.ptp(), not frame.ptp() — removed from
                              # ndarray in NumPy 2.x, function form still works
        self._prepped = ((frame - frame.min()) / (span + 1e-8)).astype(np.float32)

    def h2d_copy(self):
        if self._is_gpu_provider:
            self._device_input = ort.OrtValue.ortvalue_from_numpy(
                self._prepped, self._device, 0
            )
        else:
            # CPU provider: no device to copy to. Correctly ~0ms.
            self._device_input = self._prepped

    def inference(self):
        if self._is_gpu_provider:
            io = self.session.io_binding()
            io.bind_ortvalue_input(self.input_name, self._device_input)
            for name in self.output_names:
                io.bind_output(name, self._device)
            self.session.run_with_iobinding(io)
            self._io_binding = io
        else:
            self._outputs = self.session.run(
                self.output_names, {self.input_name: self._device_input}
            )

    def d2h_copy(self):
        if self._is_gpu_provider:
            self._outputs = self._io_binding.copy_outputs_to_cpu()
        # CPU provider: outputs are already host-side from inference(). ~0ms.

    def postprocess(self):
        """Minimal, honest postprocessing: touch the output tensor.
        Real pipelines (NMS, decode, tracking) should replace this via
        a custom stage_fns dict."""
        _ = np.asarray(self._outputs[0]).sum()

    def stage_fns(self):
        return {
            "capture": self.capture,
            "preprocess": self.preprocess,
            "h2d_copy": self.h2d_copy,
            "inference": self.inference,
            "d2h_copy": self.d2h_copy,
            "postprocess": self.postprocess,
        }
