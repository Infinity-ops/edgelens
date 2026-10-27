"""
edgelens.benchmark.image_inputs
---------------------------------
`--inputs images/`: a folder of JPEG / PNG / BMP / WebP images (or a single
image file) as real inputs for a vision model, without writing a .npy first.

    edgelens benchmark --model resnet50.onnx --inputs val/ --labels subfolders \
        --image-prep imagenet --iterations 1000

Where the work happens (so the timeline shows what a camera pipeline pays):
  before the run     decode + resize every image once (uint8, kept in RAM),
                     like frames already sitting in a capture buffer
  capture stage      copy the next frame out of that buffer
  preprocess stage   colour order, scaling / mean-std, HWC -> CHW, dtype

Layout follows the model input: [1, 3, H, W] (NCHW) or [1, H, W, 3] (NHWC);
1 channel = greyscale. H and W must be fixed (or set with --input-shape).

--image-prep: a preset, optionally followed by key=value overrides
(comma separated), e.g. `imagenet`, `yolo`, `unit,resize=crop`,
`resize=stretch,mean=0.5/0.5/0.5,std=0.5/0.5/0.5,channels=bgr`.

  preset     resize     scaling                         typical models
  unit       stretch    x / 255 -> [0, 1]   (default)   many custom models
  imagenet   crop       (x/255 - mean) / std, ImageNet   torchvision / timm classifiers
  yolo       letterbox  x / 255                         YOLO-family detectors
  signed     stretch    x / 127.5 - 1 -> [-1, 1]        MobileNet (TF), many TFLite exports
  raw        stretch    x as float, 0..255              models with normalisation inside

  resize:   stretch (to H x W), crop (shorter side to size / 0.875, centre crop:
            the torchvision evaluation transform), letterbox (fit, pad grey 114)
  channels: rgb (default) or bgr (OpenCV-trained models)
  uint8 model inputs get the resized pixels unscaled.

Labels: `--labels subfolders` takes the class from the folder each image is in
(`val/cat/1.jpg` -> class of "cat"), classes numbered in sorted folder-name
order (the torchvision ImageFolder convention: check it matches the model's
class order). Or a .npy / .csv label file in the sorted file order (the order
is printed and recorded).
"""

import hashlib
import os
from pathlib import Path

import numpy as np

from .real_inputs import InputError

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_DECODED_BYTES = 2 * 1024 ** 3      # decoded frames are kept in RAM
CROP_RATIO = 0.875                    # 224 / 256
LETTERBOX_FILL = 114

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PRESETS = {
    "unit": {"resize": "stretch", "scale": "unit"},
    "imagenet": {"resize": "crop", "scale": "unit", "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "yolo": {"resize": "letterbox", "scale": "unit"},
    "signed": {"resize": "stretch", "scale": "signed"},
    "raw": {"resize": "stretch", "scale": "raw"},
}
RESIZES = ("stretch", "crop", "letterbox")


def is_image_source(path):
    p = Path(path)
    return p.is_dir() or p.suffix.lower() in IMAGE_EXTENSIONS


def parse_image_prep(spec):
    """'imagenet,channels=bgr' -> {resize, scale, mean, std, channels, spec}."""
    spec = (spec or "unit").strip()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    prep = {"resize": "stretch", "scale": "unit", "mean": None, "std": None, "channels": "rgb"}
    if parts and "=" not in parts[0]:
        name = parts.pop(0).lower()
        if name not in PRESETS:
            raise InputError(f"--image-prep: unknown preset '{name}'; use one of "
                             f"{', '.join(PRESETS)} (optionally with key=value overrides).")
        prep.update(PRESETS[name])
    for part in parts:
        if "=" not in part:
            raise InputError(f"--image-prep: '{part}' is not key=value (the preset comes first).")
        k, v = (s.strip().lower() for s in part.split("=", 1))
        if k == "resize":
            if v not in RESIZES:
                raise InputError(f"--image-prep resize must be one of {', '.join(RESIZES)}.")
            prep["resize"] = v
        elif k == "channels":
            if v not in ("rgb", "bgr"):
                raise InputError("--image-prep channels must be rgb or bgr.")
            prep["channels"] = v
        elif k == "scale":
            if v not in ("unit", "signed", "raw"):
                raise InputError("--image-prep scale must be unit, signed or raw.")
            prep["scale"] = v
        elif k in ("mean", "std"):
            try:
                prep[k] = tuple(float(x) for x in v.split("/"))
            except ValueError:
                raise InputError(f"--image-prep {k} takes numbers separated by '/', "
                                 f"e.g. {k}=0.5/0.5/0.5.") from None
        else:
            raise InputError(f"--image-prep: unknown key '{k}' (resize, scale, mean, std, "
                             f"channels).")
    if (prep["mean"] is None) != (prep["std"] is None):
        raise InputError("--image-prep: give mean and std together.")
    prep["spec"] = spec
    return prep


def _layout(shape):
    shape = tuple(shape)
    if len(shape) != 4 or shape[0] != 1:
        raise InputError(f"--inputs with images needs a model input of [1, C, H, W] or "
                         f"[1, H, W, C]; this model takes {list(shape)}.")
    if shape[1] in (1, 3):
        return "NCHW", shape[1], shape[2], shape[3]
    if shape[3] in (1, 3):
        return "NHWC", shape[3], shape[1], shape[2]
    raise InputError(f"--inputs with images: can't tell the channel axis of {list(shape)} "
                     f"(expected 1 or 3 channels at position 1 or 3).")


def _list_images(path):
    """(files, class names or None, class index per file or None)."""
    p = Path(path)
    if p.is_file():
        return [p], None, None
    if not p.is_dir():
        raise InputError(f"--inputs: {path} does not exist.")
    top = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
    dirs = sorted(d for d in p.iterdir() if d.is_dir() and not d.name.startswith("."))
    by_class = [(d.name, sorted(f for f in d.rglob("*") if f.is_file()
                                and f.suffix.lower() in IMAGE_EXTENSIONS)) for d in dirs]
    by_class = [(c, fs) for c, fs in by_class if fs]
    if by_class and not top:
        files, idx = [], []
        for k, (_, fs) in enumerate(by_class):
            files += fs
            idx += [k] * len(fs)
        return files, [c for c, _ in by_class], np.array(idx, np.int64)
    files = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise InputError(f"--inputs: no images ({', '.join(IMAGE_EXTENSIONS)}) in {path}.")
    return files, None, None


def _resize(img, h, w, mode, Image):
    if mode == "stretch":
        return img.resize((w, h), Image.BILINEAR)
    iw, ih = img.size
    if mode == "crop":
        s = max(h / ih, w / iw) / CROP_RATIO
        nw, nh = max(w, round(iw * s)), max(h, round(ih * s))
        img = img.resize((nw, nh), Image.BILINEAR)
        left, top = (nw - w) // 2, (nh - h) // 2
        return img.crop((left, top, left + w, top + h))
    s = min(h / ih, w / iw)                                  # letterbox
    nw, nh = max(1, round(iw * s)), max(1, round(ih * s))
    canvas = Image.new(img.mode, (w, h), (LETTERBOX_FILL,) * len(img.getbands()))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((w - nw) // 2, (h - nh) // 2))
    return canvas


class ImageFolderInputs:
    """Same interface as RealInputs: .n, .sample(k), .prepare(raw), .describe()."""

    format = "images"

    def __init__(self, path, model_inputs, image_prep=None):
        try:
            from PIL import Image
        except ImportError:
            raise InputError("--inputs with images needs Pillow:  pip install pillow  "
                             "(or convert the images to a .npy yourself).") from None
        if len(model_inputs) != 1:
            raise InputError(f"--inputs with images works for single-input models; this model "
                             f"has {len(model_inputs)} inputs. Use an .npz.")
        self.path = str(path)
        self.name, shape, dtype = model_inputs[0]
        self.dtype = np.dtype(dtype)
        self.layout, self.c, self.h, self.w = _layout(shape)
        self.prep = parse_image_prep(image_prep)
        files, self.classes, self.labels = _list_images(path)
        self.n = len(files)
        need = self.n * self.h * self.w * self.c
        if need > MAX_DECODED_BYTES:
            raise InputError(f"--inputs: {self.n} images at {self.w}x{self.h} need "
                             f"{need / 1e9:.1f} GB decoded; use a subset (a folder of a few "
                             f"thousand images is plenty for latency and accuracy).")
        root = Path(path) if Path(path).is_dir() else Path(path).parent
        self.files = [str(f.relative_to(root)) for f in files]
        self.frames = np.empty((self.n, self.h, self.w, self.c), np.uint8)
        h = hashlib.sha256()
        mode = "L" if self.c == 1 else "RGB"
        for k, f in enumerate(files):
            data = Path(f).read_bytes()
            h.update(self.files[k].encode() + b"\0" + data)
            try:
                with Image.open(f) as im:
                    im = im.convert(mode)
                    arr = np.asarray(_resize(im, self.h, self.w, self.prep["resize"], Image))
            except Exception as e:                       # noqa: BLE001 - name the file
                raise InputError(f"--inputs: can't read image {f}: {e}") from e
            self.frames[k] = arr.reshape(self.h, self.w, self.c)
        self.sha256 = h.hexdigest()[:16]
        self._mean = (np.array(self.prep["mean"], np.float32).reshape(1, 1, -1)
                      if self.prep["mean"] else None)
        self._std = (np.array(self.prep["std"], np.float32).reshape(1, 1, -1)
                     if self.prep["std"] else None)
        if self._mean is not None and self._mean.size not in (1, self.c):
            raise InputError(f"--image-prep mean/std need 1 or {self.c} values.")

    def sample(self, k):
        """The frame as a receiver holds it: resized uint8 pixels (HWC)."""
        return {self.name: self.frames[k].copy()}

    def prepare(self, raw):
        """Colour order, scaling, layout, dtype: the model's preprocessing."""
        x = raw[self.name]
        if self.prep["channels"] == "bgr" and self.c == 3:
            x = x[..., ::-1]
        if np.issubdtype(self.dtype, np.integer):
            y = x
        else:
            y = x.astype(np.float32)
            if self.prep["scale"] == "unit":
                y = y / 255.0
            elif self.prep["scale"] == "signed":
                y = y / 127.5 - 1.0
            if self._mean is not None:
                y = (y - self._mean) / self._std
        if self.layout == "NCHW":
            y = y.transpose(2, 0, 1)
        return {self.name: np.ascontiguousarray(y[None], dtype=self.dtype)}

    def describe(self):
        d = {"path": self.path, "format": "images", "samples": self.n, "sha256": self.sha256,
             "shapes": {self.name: [self.h, self.w, self.c]}, "layout": self.layout,
             "image_prep": {k: v for k, v in self.prep.items() if v is not None},
             "first_files": self.files[:3]}
        if self.classes:
            d["classes"] = self.classes if len(self.classes) <= 50 else len(self.classes)
        return d


_CACHE = {}


def open_images(path, model_inputs, image_prep=None):
    """Decoded once per process (sweep opens one session per backend)."""
    key = (os.path.abspath(str(path)), tuple((n, tuple(s), str(d)) for n, s, d in model_inputs),
           image_prep or "unit")
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = ImageFolderInputs(path, model_inputs, image_prep)
    return _CACHE[key]
