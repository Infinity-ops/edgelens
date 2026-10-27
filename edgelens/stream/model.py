"""
edgelens.stream.model
-----------------------
The model of a stream spec: an ONNX file (ONNX Runtime) or a Python function
("path/to/file.py:predict" or "package.module:predict") that takes the window
[1, sequence, width] float32 and returns a number or an array.
"""

import hashlib
import importlib
import importlib.util
import os

import numpy as np

from .spec import SpecError


class Model:
    def __init__(self, fn, info, batch_fn=None):
        self._fn = fn
        self._batch = batch_fn
        self.info = info

    def __call__(self, window):
        return self._fn(window)

    def predict_many(self, wins):
        """[n, seq, width] -> [n] (batched when the model allows it)."""
        if len(wins) == 0:
            return np.zeros(0)
        if self._batch is not None:
            out = []
            for k in range(0, len(wins), 256):
                out.append(self._batch(wins[k:k + 256]))
            return np.concatenate(out)
        return np.array([self._fn(w[None]) for w in wins], dtype=np.float64)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _python_fn(ref, base_dir):
    target, _, name = ref.rpartition(":")
    if not target or not name:
        raise SpecError(f"model.python '{ref}': use 'file.py:function' or 'module:function'.")
    if target.endswith(".py"):
        path = target if os.path.isabs(target) else os.path.join(base_dir, target)
        sp = importlib.util.spec_from_file_location("edgelens_stream_model", path)
        if sp is None or not os.path.exists(path):
            raise SpecError(f"model.python: {path} not found.")
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    fn = getattr(mod, name, None)
    if not callable(fn):
        raise SpecError(f"model.python: '{name}' is not a function in {target}.")
    return fn


def _first(out, index):
    a = np.asarray(out[0] if isinstance(out, (list, tuple)) else out, dtype=np.float64)
    return float(a.reshape(-1)[index]) if a.ndim else float(a)


def load_model(spec):
    m = spec["model"]
    f = spec["features"]
    want = (f["sequence"], f["width"])
    idx = m["output_index"]
    if m["python"]:
        fn = _python_fn(m["python"], spec["base_dir"])
        return Model(lambda w: _first(fn(w), idx),
                     {"kind": "python", "ref": m["python"], "provider": "python"})
    if not m["onnx"]:
        raise SpecError("the spec has no model: add model.onnx or model.python.")
    if not os.path.exists(m["onnx"]):
        raise SpecError(f"model.onnx: {m['onnx']} not found.")
    try:
        import onnxruntime as ort
    except ImportError:
        raise SpecError("ONNX models need onnxruntime (pip install 'edgelens[onnx]'; on "
                        "Jetson NVIDIA's onnxruntime-gpu wheel).") from None
    so = ort.SessionOptions()
    if m.get("threads"):
        so.intra_op_num_threads = int(m["threads"])
    avail = ort.get_available_providers()
    prov = m["provider"]
    if prov not in avail:
        raise SpecError(f"model.provider {prov} is not available here ({', '.join(avail)}).")
    sess = ort.InferenceSession(m["onnx"], sess_options=so, providers=[prov])
    inp = sess.get_inputs()[0]
    shape = list(inp.shape)
    if len(shape) != 3:
        raise SpecError(f"{os.path.basename(m['onnx'])}: input '{inp.name}' has shape {shape}, "
                        f"expected [batch, sequence, features].")
    for got, need, what in ((shape[1], want[0], "sequence"), (shape[2], want[1], "features")):
        if isinstance(got, int) and got != need:
            hint = ""
            if what == "features":
                hint = (f" The spec gives {len(f['stats'])} stats x {len(spec['data']['columns'])} "
                        f"columns = {need}. Training code that concatenates the block's own "
                        f"row with its statistics needs 'last' in features.stats.")
            raise SpecError(f"{os.path.basename(m['onnx'])} expects {what} = {got}, the spec "
                            f"makes {need}.{hint}")
    name = inp.name
    batch_ok = not isinstance(shape[0], int)

    def one(w):
        return _first(sess.run(None, {name: np.ascontiguousarray(w, dtype=np.float32)}), idx)

    def many(ws):
        out = sess.run(None, {name: np.ascontiguousarray(ws, dtype=np.float32)})[0]
        return np.asarray(out, dtype=np.float64).reshape(len(ws), -1)[:, idx]

    info = {"kind": "onnx", "file": os.path.basename(m["onnx"]), "sha256": _sha(m["onnx"]),
            "provider": sess.get_providers()[0], "input": name, "input_shape": shape,
            "onnxruntime": ort.__version__}
    return Model(one, info, batch_fn=many if batch_ok else None)
