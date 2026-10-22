"""
edgelens.benchmark.real_inputs
--------------------------------
Real model inputs and labels for `--model` runs (benchmark, periodic, soak,
contract run): the windows a partner already prepared, fed to the model one
per iteration, and — with labels — how accurate the model was on them.

Why: `--model` used synthetic random inputs, which time a model but say
nothing about it on real data; and a model handed over as ONNX + prepared
windows (SML's motor-dropout LSTM: 2437 windows of 100 x 150, float32, plus a
regression target) needed hand-written replay scripts. For raw recordings
(CSV + a stream spec with windowing and scaling) use `edgelens replay`; this
is for windows that are already model inputs.

Formats:
  .npy   one array, first dimension = sample (memory-mapped: a large file is
         not read into RAM)
  .npz   one array per model input name (or a single array for a
         single-input model)
  .bin / .raw / .f32   raw little-endian values in the model input's dtype,
         no header, single-input model; the number of samples is the file
         size / one input's size (must divide evenly)
Each sample is the model input's shape, or that shape without its leading
batch dimension of 1 ([N, 100, 150] feeds a [1, 100, 150] input).

Labels (.npy, raw float32 .bin, or .csv/.txt, one per sample):
  one value per sample and a single-value model output -> regression:
  RMSE, MAE, worst error, and RMSE per label level when there are few
  levels; an integer label per sample and a vector output -> classification
  accuracy (argmax).

Real inputs are fed exactly as given (no normalisation): partners' windows
are usually scaled already.
"""

import hashlib
import os
from pathlib import Path

import numpy as np

RAW_EXTENSIONS = (".bin", ".raw", ".f32")
MAX_LEVELS = 20            # per-level RMSE only when the labels take this few values


class InputError(ValueError):
    pass


def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _fit_shape(name, arr, shape, path):
    shape = tuple(shape)
    if tuple(arr.shape) == shape:
        return arr[None]                                  # a single example
    if tuple(arr.shape[1:]) == shape:
        return arr
    if shape and shape[0] == 1 and tuple(arr.shape[1:]) == shape[1:]:
        return arr[:, None]                               # [N, ...] -> [N, 1, ...]
    raise InputError(
        f"{path}: input '{name}' has shape {tuple(arr.shape)}; the model takes {list(shape)}, "
        f"so the file needs [N, {', '.join(map(str, shape))}]"
        + (f" or [N, {', '.join(map(str, shape[1:]))}]" if shape and shape[0] == 1 else "") + ".")


class RealInputs:
    """Samples from a file, one dict {input name: array} per sample."""

    def __init__(self, path, model_inputs):
        """model_inputs: [(name, shape, dtype)] as OnnxStagePipeline resolves them."""
        self.path = str(path)
        ext = Path(path).suffix.lower()
        if not os.path.exists(path):
            raise InputError(f"--inputs: {path} does not exist.")
        names = [n for n, _, _ in model_inputs]
        arrays = {}
        if ext == ".npz":
            data = np.load(path, allow_pickle=False)
            for name in names:
                if name in data.files:
                    arrays[name] = data[name]
            if not arrays and len(data.files) == 1 and len(names) == 1:
                arrays[names[0]] = data[data.files[0]]
            missing = [n for n in names if n not in arrays]
            if missing:
                raise InputError(f"{path} has no array for input(s) {missing}; it holds "
                                 f"{data.files}. Name each array after its model input.")
            self.format = "npz"
        elif ext == ".npy":
            if len(names) != 1:
                raise InputError(f"{path} is one array but the model has {len(names)} inputs "
                                 f"({', '.join(names)}); use an .npz with one array per input.")
            arrays[names[0]] = np.load(path, mmap_mode="r", allow_pickle=False)
            self.format = "npy"
        elif ext in RAW_EXTENSIONS:
            if len(names) != 1:
                raise InputError(f"{path}: raw files work for single-input models only; the "
                                 f"model has {len(names)} inputs. Use an .npz.")
            name, shape, dtype = model_inputs[0]
            dt = np.dtype(dtype).newbyteorder("<")
            per = int(np.prod(shape))
            size = os.path.getsize(path)
            if size % (per * dt.itemsize):
                raise InputError(
                    f"{path}: {size} bytes is not a whole number of {list(shape)} {dt.name} "
                    f"inputs ({per * dt.itemsize} bytes each). Check the model's input shape and "
                    f"that the file holds {dt.name} values with no header.")
            arrays[name] = np.memmap(path, dtype=dt, mode="r").reshape((-1,) + tuple(shape))
            self.format = f"raw {dt.name}"
        else:
            raise InputError(f"--inputs: {path}: use .npy, .npz or raw .bin/.raw/.f32.")

        self.arrays, self.dtypes, counts = {}, {}, []
        for name, shape, dtype in model_inputs:
            a = _fit_shape(name, arrays[name], shape, path)
            self.arrays[name] = a
            self.dtypes[name] = np.dtype(dtype)
            counts.append(a.shape[0])
        if len(set(counts)) > 1:
            raise InputError(f"{path}: the inputs have different sample counts {counts}.")
        self.n = counts[0]
        if self.n == 0:
            raise InputError(f"{path} holds no samples.")
        self.sha256 = file_sha256(path)

    def sample(self, k):
        """One sample as contiguous arrays in the model's dtype: the copy is
        what a receiver would hold; reading from the memory map directly
        would count the file in the process's memory (and look like a leak)."""
        return {name: np.ascontiguousarray(a[k], dtype=self.dtypes[name])
                for name, a in self.arrays.items()}

    def describe(self):
        return {"path": self.path, "format": self.format, "samples": self.n,
                "sha256": self.sha256,
                "shapes": {n: list(a.shape[1:]) for n, a in self.arrays.items()}}


def load_labels(path, n):
    ext = Path(path).suffix.lower()
    if not os.path.exists(path):
        raise InputError(f"--labels: {path} does not exist.")
    if ext == ".npy":
        y = np.load(path, allow_pickle=False)
    elif ext in RAW_EXTENSIONS:
        y = np.fromfile(path, dtype="<f4")
    elif ext in (".csv", ".txt"):
        y = np.loadtxt(path, delimiter="," if ext == ".csv" else None, ndmin=1)
    else:
        raise InputError(f"--labels: {path}: use .npy, raw float32 .bin, or .csv/.txt.")
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[1:] == (1,) * (y.ndim - 1):
        y = y.reshape(len(y))
    if len(y) != n:
        raise InputError(f"--labels: {path} has {len(y)} labels but the inputs have {n} "
                         f"samples; there must be one label per sample, in the same order.")
    return y


INPUT_ORDERS = ("sequential", "spread")


def _coprime_stride(n):
    s = max(1, int(round(n * 0.381966)))       # ~1/phi^2: spreads well for any n
    while np.gcd(s, n) != 1:
        s += 1
    return s


class SampleOrder:
    """sequential: recording order (soaks, deployment). spread: a fixed stride
    that visits every sample once per pass in a scattered order, so a short
    run already covers the whole recording (every label level)."""

    def __init__(self, n, order="sequential"):
        if order not in INPUT_ORDERS:
            raise InputError(f"--inputs-order must be one of {', '.join(INPUT_ORDERS)}.")
        self.n, self.order = n, order
        self.stride = 1 if order == "sequential" else _coprime_stride(n)

    def index(self, i):
        return (i * self.stride) % self.n


class Predictions:
    """Preallocated: recording a prediction allocates nothing that lives on."""

    def __init__(self, n):
        self.value = np.full(n, np.nan)
        self.kind = None
        self.count = 0

    def record(self, k, outputs):
        out = np.asarray(outputs[0])
        if self.kind is None:
            self.kind = "regression" if out.size == 1 else "classification"
        self.value[k] = out.ravel()[0] if self.kind == "regression" else int(np.argmax(out))
        self.count += 1

    def quality(self, labels, labels_path=None):
        ok = np.isfinite(self.value)
        done = int(ok.sum())
        out = {"labels": labels_path, "samples": int(len(labels)), "samples_predicted": done,
               "full_pass": done == len(labels), "task": self.kind}
        if not done:
            return out
        p, y = self.value[ok], np.asarray(labels, dtype=float)[ok]
        if self.kind == "classification":
            out["accuracy"] = round(float(np.mean(p == y)), 5)
            return out
        err = p - y
        out.update(rmse=round(float(np.sqrt(np.mean(err ** 2))), 5),
                   mae=round(float(np.mean(np.abs(err))), 5),
                   max_abs_error=round(float(np.max(np.abs(err))), 5))
        levels = np.unique(y)
        if len(levels) <= MAX_LEVELS:
            out["per_level"] = []
            for lv in levels:
                m = y == lv
                out["per_level"].append({
                    "label": float(lv), "samples": int(m.sum()),
                    "rmse": round(float(np.sqrt(np.mean(err[m] ** 2))), 5),
                    "mean_prediction": round(float(p[m].mean()), 5)})
        return out
