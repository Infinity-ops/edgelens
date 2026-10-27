"""--inputs with a folder of images: decode/resize once, normalise per
--image-prep in the preprocess stage, labels from class folders.

Uses the repository's tiny classifier ([1, 3, 32, 32] -> [1, 10]) and small
generated PNGs, and checks every number against preprocessing done by hand.
"""
from pathlib import Path

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
Image = pytest.importorskip("PIL.Image")

from edgelens.benchmark import image_inputs as ii  # noqa: E402
from edgelens.benchmark.real_inputs import InputError, open_inputs  # noqa: E402
from edgelens.benchmark.runner import run_benchmark  # noqa: E402

TINY = str(Path(__file__).parent / "fixtures" / "tiny_model.onnx")
SPEC = [("input", (1, 3, 32, 32), np.float32)]
CLASSES = ("cat", "dog", "fox")


@pytest.fixture
def folder(tmp_path):
    rng = np.random.default_rng(5)
    root = tmp_path / "val"
    for c in CLASSES:
        (root / c).mkdir(parents=True)
        for k in range(4):
            px = rng.integers(0, 256, (40 + 8 * k, 48, 3), dtype=np.uint8)   # not 32x32
            Image.fromarray(px).save(root / c / f"{k}.png")
    return root


def _by_hand(path, prep, size=32):
    """Independent preprocessing: PIL resize + NumPy normalisation, NCHW."""
    p = ii.parse_image_prep(prep)
    with Image.open(path) as im:
        im = ii._resize(im.convert("RGB"), size, size, p["resize"], Image)
        x = np.asarray(im).astype(np.float32)
    if p["channels"] == "bgr":
        x = x[..., ::-1]
    x = {"unit": x / 255, "signed": x / 127.5 - 1, "raw": x}[p["scale"]]
    if p["mean"]:
        x = (x - np.array(p["mean"], np.float32)) / np.array(p["std"], np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32)


# ---- --image-prep ----

def test_presets_and_overrides():
    p = ii.parse_image_prep("imagenet")
    assert p["resize"] == "crop" and p["mean"] == ii.IMAGENET_MEAN and p["channels"] == "rgb"
    p = ii.parse_image_prep("yolo,channels=bgr")
    assert p["resize"] == "letterbox" and p["channels"] == "bgr" and p["mean"] is None
    p = ii.parse_image_prep("resize=crop,mean=0.5/0.5/0.5,std=0.5/0.5/0.5")
    assert p["scale"] == "unit" and p["std"] == (0.5, 0.5, 0.5)
    assert ii.parse_image_prep(None)["resize"] == "stretch"


@pytest.mark.parametrize("bad, msg", [("vgg", "unknown preset"), ("unit,resize=pad", "resize must"),
                                      ("mean=0.5/0.5/0.5", "mean and std together"),
                                      ("unit,colour=rgb", "unknown key")])
def test_bad_prep_is_explained(bad, msg):
    with pytest.raises(InputError, match=msg):
        ii.parse_image_prep(bad)


# ---- geometry and normalisation ----

def test_crop_and_letterbox_geometry():
    im = Image.new("RGB", (64, 32), (255, 0, 0))
    lb = np.asarray(ii._resize(im, 32, 32, "letterbox", Image))
    assert lb.shape == (32, 32, 3)
    assert (lb[:8] == 114).all() and (lb[-8:] == 114).all() and (lb[16, :, 0] == 255).all()
    cr = ii._resize(im, 32, 32, "crop", Image)
    assert cr.size == (32, 32) and (np.asarray(cr)[..., 0] == 255).all()


@pytest.mark.parametrize("prep", ["unit", "imagenet", "signed", "raw", "yolo,channels=bgr"])
def test_prepared_tensor_matches_hand_preprocessing(folder, prep):
    src = ii.ImageFolderInputs(folder, SPEC, prep)
    k = 5
    got = src.prepare(src.sample(k))["input"]
    want = _by_hand(folder / src.files[k], prep)
    assert got.shape == (1, 3, 32, 32) and got.dtype == np.float32 and got.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


def test_layouts_greyscale_and_uint8_inputs(folder):
    assert ii._layout((1, 224, 224, 3))[0] == "NHWC" and ii._layout((1, 3, 224, 224))[0] == "NCHW"
    with pytest.raises(InputError, match="channel axis"):
        ii._layout((1, 5, 32, 32))
    g = ii.ImageFolderInputs(folder, [("x", (1, 1, 16, 16), np.float32)])
    assert g.prepare(g.sample(0))["x"].shape == (1, 1, 16, 16)
    u = ii.ImageFolderInputs(folder, [("x", (1, 16, 16, 3), np.uint8)], "unit")
    out = u.prepare(u.sample(0))["x"]
    assert out.dtype == np.uint8 and out.shape == (1, 16, 16, 3) and out.max() > 1


# ---- folders and labels ----

def test_class_folders_give_labels_in_sorted_order(folder):
    src = ii.ImageFolderInputs(folder, SPEC)
    assert src.classes == list(CLASSES) and src.n == 12
    assert list(src.labels) == [0] * 4 + [1] * 4 + [2] * 4
    assert src.files[4].startswith("dog")
    d = src.describe()
    assert d["format"] == "images" and d["image_prep"]["resize"] == "stretch" and d["sha256"]


def test_changing_one_image_changes_the_fingerprint(folder):
    a = ii.ImageFolderInputs(folder, SPEC).sha256
    Image.new("RGB", (8, 8)).save(folder / "cat" / "0.png")
    assert ii.ImageFolderInputs(folder, SPEC).sha256 != a


def test_single_image_file(folder):
    src = open_inputs(str(folder / "dog" / "1.png"), SPEC)
    assert src.n == 1 and src.classes is None


def test_wrong_combinations_are_refused(folder, tmp_path):
    np.save(tmp_path / "x.npy", np.zeros((2, 3, 32, 32), np.float32))
    with pytest.raises(InputError, match="applies to --inputs with images"):
        open_inputs(str(tmp_path / "x.npy"), SPEC, "imagenet")
    flat = tmp_path / "flat"
    flat.mkdir()
    Image.new("RGB", (8, 8)).save(flat / "a.png")
    with pytest.raises(RuntimeError, match="folder of class folders"):
        run_benchmark(iterations=1, model_path=TINY, inputs=str(flat), labels="subfolders")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="no images"):
        run_benchmark(iterations=1, model_path=TINY, inputs=str(empty))
    with pytest.raises(RuntimeError, match="--image-prep needs --inputs"):
        run_benchmark(iterations=1, model_path=TINY, image_prep="imagenet")


# ---- end to end ----

def test_benchmark_accuracy_matches_an_independent_loop(folder):
    r = run_benchmark(iterations=12, warmup=0, model_path=TINY, inputs=str(folder),
                      labels="subfolders", image_prep="imagenet")
    q = r["quality"]
    s = ort.InferenceSession(TINY, providers=["CPUExecutionProvider"])
    files = sorted(folder.rglob("*.png"))
    pred = np.array([int(np.argmax(s.run(None, {"input": _by_hand(f, "imagenet")})[0]))
                     for f in files])
    truth = np.repeat(np.arange(3), 4)
    assert q["task"] == "classification" and q["full_pass"]
    assert q["accuracy"] == pytest.approx(np.mean(pred == truth), abs=1e-5)
    cfg = r["pipeline"]["config"]["inputs"]
    assert cfg["format"] == "images" and cfg["classes"] == list(CLASSES)
    assert cfg["image_prep"]["spec"] == "imagenet"


def test_sweep_uses_the_same_prepared_images(folder):
    from edgelens.sweep import run_sweep
    summary, results = run_sweep(TINY, ["cpu"], iterations=5, inputs_path=str(folder),
                                 eval_samples=12, labels_path="subfolders", image_prep="imagenet")
    single = run_benchmark(iterations=12, warmup=0, model_path=TINY, inputs=str(folder),
                           labels="subfolders", image_prep="imagenet")["quality"]
    assert summary["backends"][0]["accuracy"] == single["accuracy"]
