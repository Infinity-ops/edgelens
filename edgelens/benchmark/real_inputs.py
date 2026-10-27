"""
edgelens.benchmark.real_inputs
--------------------------------
Real model inputs and labels for `--model` runs (benchmark, periodic, soak,
contract run, sweep): samples you already prepared (images, sensor windows,
feature vectors), fed to the model one per iteration, and, with labels, how
accurate the model was on them on this board and backend.

Why: synthetic random inputs time a model but say nothing about what it
answers; and an accuracy number measured on a workstation says nothing about
the quantised / TensorRT / FP16 build on the device. For raw recordings
(CSV + a stream spec with windowing and scaling) use `edgelens replay`; this
is for data that is already model input.

Formats:
  .npy   one array, first dimension = sample (memory-mapped: a large file is
         not read into RAM)
  .npz   one array per model input name (or a single array for a
         single-input model)
  .bin / .raw / .f32   raw little-endian values in the model input's dtype,
         no header, single-input model; the number of samples is the file
         size / one input's size (must divide evenly)
Each sample is the model input's shape, or that shape without its leading
batch dimension of 1 ([N, 3, 224, 224] feeds a [1, 3, 224, 224] input).
Inputs are fed exactly as given (no normalisation, no resizing): prepare them
the way the model was trained.

Labels (.npy, raw float32 .bin, or .csv/.txt), one row per sample, scored
against the model's FIRST output:
  [N]  + a single output value   regression: RMSE, MAE, worst error, RMSE
                                 per label level (few distinct labels); with
                                 0/1 labels also accuracy, precision, recall
                                 at > 0.5 (binary detectors)
  [N, k] + k output values       multi-value regression: RMSE overall and
                                 per output
  [N] class indices + C outputs  classification: top-1 (argmax), top-5
                                 (C > 5) and per-class accuracy
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

    def prepare(self, raw):
        """Real arrays are model inputs already: fed as given."""
        return raw

    def describe(self):
        return {"path": self.path, "format": self.format, "samples": self.n,
                "sha256": self.sha256,
                "shapes": {n: list(a.shape[1:]) for n, a in self.arrays.items()}}


def open_inputs(path, model_inputs, image_prep=None):
    """RealInputs for arrays, ImageFolderInputs for a folder or an image file."""
    from .image_inputs import is_image_source, open_images
    if is_image_source(path):
        return open_images(path, model_inputs, image_prep)
    if image_prep:
        raise InputError("--image-prep applies to --inputs with images (a folder or an image "
                         "file); array files are fed as given.")
    return RealInputs(path, model_inputs)


SUBFOLDERS = "subfolders"


def resolve_labels(labels, inputs):
    """--labels FILE, or --labels subfolders (the class folder of each image)."""
    if labels == SUBFOLDERS:
        if getattr(inputs, "labels", None) is None:
            raise InputError("--labels subfolders needs --inputs pointing at a folder of class "
                             "folders (images/cat/1.jpg, images/dog/2.jpg, ...).")
        return inputs.labels
    return load_labels(labels, inputs.n)


def load_labels(path, n):
    ext = Path(path).suffix.lower()
    if not os.path.exists(path):
        raise InputError(f"--labels: {path} does not exist.")
    if ext == ".npy":
        y = np.load(path, allow_pickle=False)
    elif ext in RAW_EXTENSIONS:
        y = np.fromfile(path, dtype="<f4")
        if len(y) != n and n and len(y) % n == 0:
            y = y.reshape(n, -1)                  # raw [N, k] values, row-major
    elif ext in (".csv", ".txt"):
        y = np.loadtxt(path, delimiter="," if ext == ".csv" else None, ndmin=1)
    else:
        raise InputError(f"--labels: {path}: use .npy, raw float32 .bin, or .csv/.txt.")
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[1:] == (1,) * (y.ndim - 1):
        y = y.reshape(len(y))
    if y.ndim > 2:
        raise InputError(f"--labels: {path} has shape {y.shape}; use [N] (one value or class "
                         f"per sample) or [N, k] (k values per sample).")
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


VECTOR_MAX = 64           # outputs up to this size are kept whole (regression vectors, small class sets)
TOP_K = 5
BINARY_THRESHOLD = 0.5


def _task(y, size):
    """What the labels and the model's first output say the task is.

      labels [N]     + 1 output value            -> regression (and, when every
                                                    label is 0 or 1, binary
                                                    accuracy at > 0.5 as well)
      labels [N, k]  + k output values           -> multi-value regression
      labels [N] integers + C > 1 output values  -> classification (argmax,
                                                    top-5 when C > 5)
    """
    y = np.asarray(y)
    if y.ndim == 2:
        if y.shape[1] != size:
            raise InputError(f"--labels have {y.shape[1]} values per sample but the model's "
                             f"first output has {size}; they must match for multi-value "
                             f"regression.")
        return "regression"
    if size == 1:
        return "binary" if np.isin(np.unique(y), (0, 1)).all() else "regression"
    if not np.all(np.mod(y, 1) == 0) or y.min() < 0 or y.max() >= size:
        raise InputError(f"--labels: the model's first output has {size} values, so labels must "
                         f"be class indices 0..{size - 1} (one per sample), or [N, {size}] "
                         f"arrays for multi-value regression.")
    return "classification"


class Predictions:
    """The model's first output per sample, preallocated (recording allocates
    nothing that lives on). Outputs up to VECTOR_MAX values are kept whole;
    larger ones (e.g. 1000 ImageNet classes) keep their top-5 class indices."""

    def __init__(self, n, output_name=None):
        self.n, self.output = n, output_name
        self.size = None
        self.vec = None            # [n, size] float32
        self.top = None            # [n, TOP_K] int32, best first
        self.count = 0

    @property
    def kind(self):                 # backward compatible: regression | classification
        return None if self.size is None else ("regression" if self.size == 1 else
                                               "classification")

    def record(self, k, outputs):
        out = np.asarray(outputs[0]).ravel()
        if self.size is None:
            self.size = out.size
            if self.size <= VECTOR_MAX:
                self.vec = np.full((self.n, self.size), np.nan, np.float32)
            else:
                self.top = np.full((self.n, TOP_K), -1, np.int32)
        if self.vec is not None:
            self.vec[k] = out
        else:
            idx = np.argpartition(-out, TOP_K)[:TOP_K]
            self.top[k] = idx[np.argsort(-out[idx])]
        self.count += 1

    def _done(self):
        if self.vec is not None:
            return np.isfinite(self.vec).all(axis=1)
        if self.top is not None:
            return self.top[:, 0] >= 0
        return np.zeros(self.n, bool)

    def quality(self, labels, labels_path=None):
        y_all = np.asarray(labels)
        ok = self._done()
        done = int(ok.sum())
        out = {"labels": labels_path, "output": self.output, "samples": int(len(y_all)),
               "samples_predicted": done, "full_pass": done == len(y_all), "task": None}
        if not done:
            return out
        try:
            task = _task(y_all, self.size)
        except InputError as e:
            out["error"] = str(e)
            return out
        out["task"] = task
        y = y_all[ok]
        if task == "classification":
            return self._classification(out, y, ok)
        p = self.vec[ok].astype(float)
        yv = y.astype(float).reshape(len(y), -1)
        err = p - yv
        out.update(rmse=round(float(np.sqrt(np.mean(err ** 2))), 5),
                   mae=round(float(np.mean(np.abs(err))), 5),
                   max_abs_error=round(float(np.max(np.abs(err))), 5))
        if yv.shape[1] > 1:
            out["outputs"] = yv.shape[1]
            out["per_output_rmse"] = [round(float(v), 5) for v in np.sqrt(np.mean(err ** 2, 0))]
            return out
        p1, y1, e1 = p[:, 0], yv[:, 0], err[:, 0]
        if task == "binary":
            pred = p1 > BINARY_THRESHOLD
            truth = y1 > BINARY_THRESHOLD
            tp, fp = int((pred & truth).sum()), int((pred & ~truth).sum())
            fn = int((~pred & truth).sum())
            out.update(threshold=BINARY_THRESHOLD,
                       accuracy=round(float(np.mean(pred == truth)), 5),
                       precision=round(tp / (tp + fp), 5) if tp + fp else None,
                       recall=round(tp / (tp + fn), 5) if tp + fn else None,
                       false_positives=fp, false_negatives=fn)
        levels = np.unique(y1)
        if len(levels) <= MAX_LEVELS:
            out["per_level"] = []
            for lv in levels:
                m = y1 == lv
                out["per_level"].append({
                    "label": float(lv), "samples": int(m.sum()),
                    "rmse": round(float(np.sqrt(np.mean(e1[m] ** 2))), 5),
                    "mean_prediction": round(float(p1[m].mean()), 5)})
        return out

    def _classification(self, out, y, ok):
        y = y.astype(int)
        if self.vec is not None:
            scores = self.vec[ok]
            top = np.argsort(-scores, axis=1)[:, :TOP_K]
        else:
            top = self.top[ok]
        out["classes"] = int(self.size)
        out["accuracy"] = round(float(np.mean(top[:, 0] == y)), 5)
        if self.size > TOP_K:
            out["top5_accuracy"] = round(float(np.mean((top == y[:, None]).any(axis=1))), 5)
        classes = np.unique(y)
        if len(classes) <= MAX_LEVELS:
            out["per_class"] = [{"class": int(c), "samples": int((y == c).sum()),
                                 "accuracy": round(float(np.mean(top[y == c, 0] == c)), 5)}
                                for c in classes]
        return out
