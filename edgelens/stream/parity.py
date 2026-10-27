"""
edgelens.stream.parity
------------------------
Train/serve parity: are the numbers the deployed pipeline feeds the model the
numbers the model was trained and evaluated on?

    edgelens parity spec.yaml scenario.csv                         (self-check)
    edgelens parity spec.yaml scenario.csv --reference x_test.bin  (training's tensor)
    edgelens parity spec.yaml scenario.csv --reference-outputs y_pred.bin
                                                     (another runtime's outputs, e.g. C++)

The reference tensor is the training-side model input for the same recording:
a .npy file, or raw float32 (.bin, C order) of shape [steps, sequence, width].
Reference outputs are one float32 per step.

Typical bugs this finds, and the signature it reports for each:
  * std with ddof=0 on one side and ddof=1 on the other (pandas vs NumPy):
    only the std features differ, by a constant ratio sqrt((W-1)/W)
  * a scaler fitted on other data, or applied to other columns: every stat of
    a column is off, by a constant
  * windows shifted by one step (padding or block alignment differ)
  * a different number of steps (rows dropped, a partial block kept)
"""

import numpy as np

from .features import StreamFeaturizer, batch_features, feature_names, windows


def _load_tensor(path, seq, width):
    if str(path).endswith(".npy"):
        a = np.load(path)
    else:
        a = np.fromfile(path, dtype=np.float32)
        per = seq * width
        if a.size % per:
            raise ValueError(f"{path}: {a.size} float32 values are not a whole number of "
                             f"[{seq}, {width}] windows (wrong spec, or not float32?)")
        a = a.reshape(-1, seq, width)
    if a.ndim != 3 or a.shape[1:] != (seq, width):
        raise ValueError(f"{path}: shape {a.shape}, the spec makes [steps, {seq}, {width}]")
    return a.astype(np.float64)


def self_check(rec, spec, steps=300):
    """Batch reference vs the streaming implementations (incremental and
    recompute) on the first `steps` model steps."""
    F, _ = batch_features(rec, spec)
    ref = windows(F, spec)[:steps]
    W = spec["features"]["block"]
    out = {}
    for mode in ("incremental", "recompute"):
        fz = StreamFeaturizer(spec, mode=mode)
        worst, k = 0.0, 0
        for b in range(min(steps, len(F))):
            w = fz.push(rec.x[b * W:(b + 1) * W])
            if w is None:
                continue
            j = b if spec["features"]["padding"] == "zeros" else b - spec["features"]["sequence"] + 1
            worst = max(worst, float(np.max(np.abs(w[0] - ref[j]))))
            k += 1
        out[mode] = {"steps": k, "max_abs_error": worst}
    return out


def compare_tensor(ours, ref, spec):
    names = feature_names(spec)
    n = min(len(ours), len(ref))
    rep = {"steps_ours": int(len(ours)), "steps_reference": int(len(ref)), "compared": n}
    if n == 0:
        rep["verdict"] = "nothing to compare"
        return rep
    # compare the newest step of each window (every block appears there once)
    a, b = ours[:n, -1, :], ref[:n, -1, :]
    err = np.abs(a - b)
    tol = 1e-4 * (1.0 + np.abs(b))
    bad = err > tol
    rep["max_abs_error"] = float(np.nanmax(err))
    rep["mismatching_values_pct"] = round(100.0 * bad.mean(), 4)
    per = np.nanmax(err, axis=0)
    order = np.argsort(-per)[:8]
    rep["worst_features"] = [{"feature": names[i], "max_abs_error": float(per[i])}
                             for i in order if per[i] > 0]
    rows = np.nonzero(bad.any(axis=1))[0]
    rep["first_mismatch_step"] = int(rows[0]) if len(rows) else None
    rep["signatures"] = _signatures(a, b, bad, spec, ours, ref)
    rep["verdict"] = "match" if not bad.any() and len(ours) == len(ref) else "MISMATCH"
    return rep


def _signatures(a, b, bad, spec, ours, ref):
    sig = []
    if len(ours) != len(ref):
        sig.append(f"different number of steps ({len(ours)} vs {len(ref)}): rows dropped, "
                   f"or a partial last block kept on one side")
    if not bad.any():
        return sig
    stats, C = spec["features"]["stats"], len(spec["data"]["columns"])
    major = spec["features"]["layout"] == "stat-major"

    def cols_of(stat_i):
        return [stat_i * C + c for c in range(C)] if major else \
            [c * len(stats) + stat_i for c in range(C)]
    bad_stats = [s for i, s in enumerate(stats) if bad[:, cols_of(i)].any()]
    if bad_stats == ["std"]:
        i = cols_of(stats.index("std"))
        x, y = a[:, i].ravel(), b[:, i].ravel()
        m = (np.abs(x) > 1e-9) & (np.abs(y) > 1e-9)
        if m.any():
            r = float(np.median(y[m] / x[m]))
            W = spec["features"]["block"]
            if abs(r - np.sqrt((W - 1) / W)) < 1e-3:
                sig.append(f"only std differs, reference/ours = {r:.4f} = sqrt((W-1)/W): the "
                           f"reference uses ddof=0 (NumPy default), this spec ddof=1 (pandas)")
            elif abs(r - np.sqrt(W / (W - 1))) < 1e-3:
                sig.append(f"only std differs, reference/ours = {r:.4f} = sqrt(W/(W-1)): the "
                           f"reference uses ddof=1 (pandas), this side ddof=0")
            else:
                sig.append(f"only the std features differ (median ratio {r:.4f})")
    for c, name in enumerate(spec["data"]["columns"]):
        idx = [(i * C + c) if major else (c * len(stats) + i) for i in range(len(stats))]
        if all(bad[:, j].any() for j in idx):
            sig.append(f"every statistic of '{name}' differs: scaler or column order "
                       f"(is the scaler fitted on the same data, for this column?)")
            if len(sig) > 6:
                break
    if len(ours) > 1 and len(ref) > 1:
        n = min(len(ours), len(ref)) - 1
        for shift, (x, y) in ((1, (ours[1:n + 1, -1], ref[:n, -1])),
                              (-1, (ours[:n, -1], ref[1:n + 1, -1]))):
            if np.allclose(x, y, rtol=1e-4, atol=1e-4):
                sig.append(f"the windows match when shifted by {shift} step: block "
                           f"alignment or padding differs by one step")
    return sig


def compare_outputs(raw, ref_path, spec):
    ref = np.fromfile(ref_path, dtype=np.float32).astype(np.float64) if not \
        str(ref_path).endswith(".npy") else np.load(ref_path).astype(np.float64).ravel()
    n = min(len(raw), len(ref))
    d = np.abs(np.asarray(raw[:n]) - ref[:n])
    rep = {"steps_ours": len(raw), "steps_reference": int(len(ref)),
           "max_abs_error": float(np.nanmax(d)) if n else None,
           "mean_abs_error": float(np.nanmean(d)) if n else None}
    dec = spec.get("decision")
    if dec and n:
        thr = dec["threshold"]
        a = np.asarray(raw[:n]) <= thr if dec["alarm_when"] == "below" else np.asarray(raw[:n]) >= thr
        b = ref[:n] <= thr if dec["alarm_when"] == "below" else ref[:n] >= thr
        rep["decisions_flipped"] = int((a != b).sum())
    return rep
