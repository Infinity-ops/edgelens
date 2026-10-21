"""
edgelens.contract
-------------------
Assumption contracts: requirements as code (v0.2 Layer 3, E1).

A contract says what a system guarantees AND under which conditions the
guarantee is claimed:

    contract: dem2-anomaly-detection
    version: "1.0"
    requirements_doc: EdgeAI-Trust D1.1
    min_duration: 30m
    min_iterations: 1000
    assumes:
      board: AGX Orin
      power_mode: MAXN
      clocks_pinned: true
      input_period_ms: {value: 1000, tolerance_pct: 5}
      max_start_temp_c: 50
      max_background_cpu_pct: 10
    guarantees:
      p99_ms:          {limit: 3000, id: D1.1-R07}
      max_miss_ratio:  {limit: 0, deadline_ms: 3000, id: D1.1-R08}
      max_energy_per_period_mj: {limit: 15000, id: D1.1-R12}
      max_temp_c:      {limit: 85, id: D1.1-R14}

Verdict (assume-guarantee semantics: A -> G):

* Every assumption HOLDS and every evidence condition (min_duration,
  min_iterations) is met: the verdict is the guarantees' verdict
  (PASS / FAIL / INCONCLUSIVE, the same rules as `edgelens validate`).
* Any assumption VIOLATED or NOT_VERIFIED, or too little evidence:
  INCONCLUSIVE, never PASS. A FAIL observed outside the assumptions is
  reported, but it is not evidence against the contract either: the run
  measured a different situation than the one the contract is about.

A typo in a contract must never weaken it silently, so unknown keys are
errors, not ignored.
"""

import difflib
import hashlib
import json
import math
import os

from .validate.engine import (FAIL, INCONCLUSIVE, NOT_MEASURED, PASS, REQUIREMENT_KEYS,
                              validate)

HOLDS, VIOLATED, NOT_VERIFIED = "HOLDS", "VIOLATED", "NOT_VERIFIED"
MET, NOT_MET = "MET", "NOT_MET"
DEFAULT_PERIOD_TOLERANCE_PCT = 5.0

TOP_KEYS = ("contract", "version", "description", "requirements_doc", "min_duration",
            "min_iterations", "assumes", "guarantees")

# assumption -> (label, extra keys allowed in its mapping form)
ASSUMPTIONS = {
    "board": ("Board", ()),
    "power_mode": ("Power mode (nvpmodel)", ()),
    "clocks_pinned": ("Clocks pinned (CPU and GPU min = max)", ()),
    "input_period_ms": ("Input period", ("tolerance_pct",)),
    "max_start_temp_c": ("Board temperature at start", ()),
    "max_temp_c": ("Thermal envelope (peak temperature during the run)", ()),
    "max_background_cpu_pct": ("Background CPU load (other processes)", ()),
    "provider": ("ONNX Runtime execution provider", ()),
    "model_sha256": ("Model file (SHA-256)", ()),
    # v0.2 Layer 4 (edgelens replay / detect)
    "min_in_envelope_pct": ("Inputs inside the training envelope", ()),
    "stream_spec_sha256": ("Stream spec (SHA-256)", ()),
}

GUARANTEE_EXTRA = {"max_miss_ratio": ("deadline_ms",)}


class ContractError(ValueError):
    pass


# ---------------------------------------------------------------- loading

def _suggest(key, known):
    near = difflib.get_close_matches(key, known, n=1)
    return f" Did you mean '{near[0]}'?" if near else ""


def _check_keys(where, obj, known):
    for k in obj:
        if k not in known:
            raise ContractError(f"Unknown key '{k}' in {where}.{_suggest(k, known)} "
                                f"Known: {', '.join(known)}")


def _spec(section, key, raw, extra=()):
    """Scalar or {value|limit: X, id, note, ...} -> {"value", "id", "note", ...}."""
    if isinstance(raw, dict):
        allowed = ("value", "limit", "id", "note") + tuple(extra)
        _check_keys(f"{section}.{key}", raw, allowed)
        if ("value" in raw) == ("limit" in raw):
            raise ContractError(f"{section}.{key}: give exactly one of 'value' or 'limit'.")
        out = {"value": raw["value"] if "value" in raw else raw["limit"],
               "id": raw.get("id"), "note": raw.get("note")}
        for e in extra:
            if e in raw:
                out[e] = raw[e]
        return out
    if isinstance(raw, list):
        raise ContractError(f"{section}.{key}: a list is not a value (use a number, text or "
                            f"true/false).")
    return {"value": raw, "id": None, "note": None}


def _number(where, v, allow_bool=False):
    if isinstance(v, bool) and not allow_bool:
        raise ContractError(f"{where}: expected a number, got {v!r}.")
    if not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)):
        raise ContractError(f"{where}: expected a number, got {v!r}.")
    return float(v)


def parse(doc, source="<contract>"):
    """Validate a contract document (dict) and return the normalized form."""
    if not isinstance(doc, dict):
        raise ContractError(f"{source}: a contract is a mapping (key: value), got "
                            f"{type(doc).__name__}.")
    _check_keys("the contract", doc, TOP_KEYS)
    if not doc.get("contract"):
        raise ContractError(f"{source}: 'contract: <name>' is required.")
    guarantees = doc.get("guarantees") or {}
    if not isinstance(guarantees, dict) or not guarantees:
        raise ContractError(f"{source}: 'guarantees' must list at least one requirement, "
                            f"e.g. p99_ms: 3000.")
    assumes = doc.get("assumes") or {}
    if not isinstance(assumes, dict):
        raise ContractError(f"{source}: 'assumes' must be a mapping.")
    _check_keys("assumes", assumes, tuple(ASSUMPTIONS))
    _check_keys("guarantees", guarantees, REQUIREMENT_KEYS)

    out = {"contract": str(doc["contract"]), "version": doc.get("version"),
           "description": doc.get("description"),
           "requirements_doc": doc.get("requirements_doc"),
           "min_duration_s": None, "min_iterations": None, "assumes": {}, "guarantees": {}}
    if doc.get("min_duration") is not None:
        from .soak import parse_duration
        try:
            out["min_duration_s"] = parse_duration(doc["min_duration"])
        except ValueError as e:
            raise ContractError(f"min_duration: {e}") from None
    if doc.get("min_iterations") is not None:
        v = doc["min_iterations"]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ContractError("min_iterations: expected a whole number >= 1.")
        out["min_iterations"] = v

    for k, raw in assumes.items():
        s = _spec("assumes", k, raw, ASSUMPTIONS[k][1])
        v = s["value"]
        if k == "clocks_pinned" and not isinstance(v, bool):
            raise ContractError("assumes.clocks_pinned: expected true or false.")
        if k in ("input_period_ms", "max_start_temp_c", "max_temp_c", "max_background_cpu_pct",
                 "min_in_envelope_pct"):
            s["value"] = _number(f"assumes.{k}", v)
            if k == "input_period_ms" and s["value"] <= 0:
                raise ContractError("assumes.input_period_ms must be > 0.")
        if k == "input_period_ms":
            s["tolerance_pct"] = _number("assumes.input_period_ms.tolerance_pct",
                                         s.get("tolerance_pct", DEFAULT_PERIOD_TOLERANCE_PCT))
        if k in ("board", "provider", "model_sha256", "stream_spec_sha256") and \
                not isinstance(v, str):
            raise ContractError(f"assumes.{k}: expected text, got {v!r}.")
        if k == "power_mode" and not isinstance(v, (str, int)):
            raise ContractError("assumes.power_mode: expected a mode name (MAXN) or id (0).")
        out["assumes"][k] = s

    for k, raw in guarantees.items():
        s = _spec("guarantees", k, raw, GUARANTEE_EXTRA.get(k, ()))
        s["value"] = _number(f"guarantees.{k}", s["value"])
        if "deadline_ms" in s:
            s["deadline_ms"] = _number(f"guarantees.{k}.deadline_ms", s["deadline_ms"])
        out["guarantees"][k] = s
    return out


def load(path):
    """Read a .yaml/.yml/.json contract file. Returns (contract, sha256)."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise ContractError(f"Cannot read contract {path}: {e.strerror or e}") from None
    text = raw.decode("utf-8")
    if str(path).lower().endswith(".json"):
        try:
            doc = json.loads(text)
        except ValueError as e:
            raise ContractError(f"{path} is not valid JSON: {e}") from None
    else:
        try:
            import yaml
        except ImportError:           # pragma: no cover - pyyaml is a dependency
            raise ContractError("Reading YAML contracts needs PyYAML: pip install pyyaml "
                                "(or write the contract as .json).") from None
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ContractError(f"{path} is not valid YAML: {e}") from None
    c = parse(doc, source=str(path))
    c["file"] = os.path.basename(str(path))
    return c, hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------- measurements

def run_duration_s(result):
    """How long the measured run lasted (seconds), or None."""
    soak = result.get("soak") or {}
    if soak.get("duration_s"):
        return float(soak["duration_s"])
    # The run's own window (first release / first span to the last end) when it
    # was measured: the telemetry span is shorter by up to one sampling interval
    # (a 1.0 s run measured as 0.9 s on a laptop failed min_duration: 1s).
    fps, n = result.get("fps"), result.get("iterations")
    if fps and n and str(result.get("throughput_basis", "")).startswith("observed"):
        return n / fps
    tel = result.get("telemetry") or {}
    if tel.get("duration_s"):
        return float(tel["duration_s"])
    if fps and n:
        return n / fps
    return None


def start_temp_c(result):
    tel = result.get("telemetry") or {}
    if tel.get("start_temp_c") is not None:
        return tel["start_temp_c"]
    for s in result.get("telemetry_series") or []:
        if s.get("max_temp_c") is not None:
            return s["max_temp_c"]
    return None


def _provider(result):
    b = result.get("backend") or {}
    if b.get("provider"):
        return b["provider"]
    return ((result.get("pipeline") or {}).get("config") or {}).get("provider")


def _declared_period(result):
    """(period_ms, how) the inputs were assumed/configured to arrive at."""
    per = result.get("periodic")
    if per and (per.get("primary") or {}).get("period_ms"):
        return per["primary"]["period_ms"], "periodic run"
    req = result.get("requirements") or {}
    p = req.get("period_ms")
    mode = result.get("mode")
    observed = mode in ("markers", "attach", "watch", "import-log") or bool(result.get("markers"))
    if p is None:
        # EdgeLens drove the loop with no period at all: back to back, which is
        # a different input situation, not an unknown one
        if mode == "hardware" and not observed and req.get("paced") is False:
            return None, "unpaced"
        return None, None
    if observed:
        return p, "declared with --period-ms"
    if req.get("paced"):
        return p, "paced run (inputs released every period)"
    return p, "unpaced"


UNITS = {"max_start_temp_c": "C", "max_temp_c": "C", "max_background_cpu_pct": "%",
         "min_in_envelope_pct": "%"}


def _row(key, label, expected, measured, status, reason=None, basis=None, spec=None):
    return {"assumption": key, "label": label, "expected": expected, "measured": measured,
            "unit": UNITS.get(key), "status": status, "reason": reason, "basis": basis,
            "id": (spec or {}).get("id"), "note": (spec or {}).get("note")}


def check_assumption(result, key, spec):
    label = ASSUMPTIONS[key][0]
    want = spec["value"]
    env = result.get("environment") or {}

    if key == "board":
        board = env.get("board")
        if not board:
            return _row(key, label, want, None, NOT_VERIFIED, "the result has no board "
                        "information", spec=spec)
        # The model string is the carrier's own on custom boards ("Taipur
        # Orin"); the module name comes from the part number in the device
        # tree ("Jetson AGX Orin 32GB"). Either may carry the wanted name.
        module = env.get("module")
        shown = f"{board} ({module})" if module and module not in board else board
        ok = want.lower() in board.lower() or bool(module and want.lower() in module.lower())
        return _row(key, label, want, shown, HOLDS if ok else VIOLATED,
                    None if ok else f"measured on '{shown}'", "environment", spec)

    if key == "power_mode":
        pm = env.get("power_mode") or {}
        if not pm or (pm.get("name") is None and pm.get("id") is None):
            from .hardware.platform_hints import power_mode_unreadable_reason
            return _row(key, label, want, None, NOT_VERIFIED,
                        power_mode_unreadable_reason(bool(env.get("is_jetson"))), spec=spec)
        got = pm.get("name") if isinstance(want, str) else pm.get("id")
        shown = pm.get("name") if pm.get("name") is not None else pm.get("id")
        if got is None:
            return _row(key, label, want, shown, NOT_VERIFIED,
                        "only the mode " + ("id" if isinstance(want, str) else "name") +
                        " is known; give the contract the other form", spec=spec)
        ok = (str(got).strip().lower() == str(want).strip().lower()
              if isinstance(want, str) else int(got) == int(want))
        return _row(key, label, want, shown, HOLDS if ok else VIOLATED,
                    None if ok else f"the run used power mode {shown}", "environment", spec)

    if key == "clocks_pinned":
        cpu, gpu = env.get("cpu_clocks_locked"), env.get("gpu_clocks_locked")
        measured = {"cpu": cpu, "gpu": gpu}
        if cpu is None or gpu is None:
            known_wrong = (cpu is not None and cpu != want) or (gpu is not None and gpu != want)
            if known_wrong:
                return _row(key, label, want, measured, VIOLATED,
                            "CPU or GPU clocks are not " + ("pinned" if want else "free"),
                            "environment", spec)
            return _row(key, label, want, measured, NOT_VERIFIED,
                        "clock state not readable for " +
                        " and ".join(n for n, v in (("CPU", cpu), ("GPU", gpu)) if v is None),
                        spec=spec)
        ok = cpu == want and gpu == want
        from .hardware.platform_hints import jetson_clocks_available, pin_clocks_fix
        pin = ("run `sudo jetson_clocks` before measuring" if jetson_clocks_available()
               else "pin clocks before measuring: " + pin_clocks_fix())
        return _row(key, label, want, measured, HOLDS if ok else VIOLATED,
                    None if ok else (pin if want else
                                     "clocks were pinned; restore with jetson_clocks --restore"),
                    "environment", spec)

    if key == "input_period_ms":
        tol = spec.get("tolerance_pct", DEFAULT_PERIOD_TOLERANCE_PCT) / 100.0
        declared, how = _declared_period(result)
        exp = f"{want:g} ms ± {tol * 100:g}%"
        if how == "unpaced":
            return _row(key, label, exp, "back-to-back", VIOLATED,
                        "inputs were run back to back, not every period: add --pace "
                        "(or use `edgelens periodic` / `edgelens contract run`)",
                        "configuration", spec)
        if declared is None:
            return _row(key, label, exp, None, NOT_VERIFIED,
                        "the run has no input period: benchmark with --period-ms --pace, "
                        "`edgelens periodic`, or record with --period-ms", spec=spec)
        if abs(declared - want) > want * tol:
            return _row(key, label, exp, f"{declared:g} ms — {how}", VIOLATED,
                        f"the run's period is {declared:g} ms", "configuration", spec)
        if how != "declared with --period-ms":
            # EdgeLens released the inputs itself on this schedule: it holds by
            # construction (keeping up with it is a guarantee, not an assumption).
            return _row(key, label, exp, f"{declared:g} ms — {how}", HOLDS, None,
                        "configuration", spec)
        fps = result.get("fps") if str(result.get("throughput_basis", "")).startswith(
            "observed") else None          # 1/mean latency says nothing about arrivals
        if not fps:
            return _row(key, label, exp, f"{declared:g} ms — {how}", HOLDS,
                        "no observed rate to cross-check", "declared", spec)
        observed = 1000.0 / fps
        shown = f"{observed:.4g} ms observed — {how}"
        if observed < want * (1 - tol):
            return _row(key, label, exp, shown, VIOLATED,
                        f"inputs arrived faster than assumed ({observed:.4g} ms)",
                        "measured", spec)
        if observed > want * (1 + tol):
            lat = result.get("latency") or {}
            saturated = ((lat.get("p50") or 0) >= want * (1 - tol)) or \
                ((result.get("deadline") or {}).get("misses") or 0) > 0
            if saturated:
                return _row(key, label, exp, shown, HOLDS,
                            "the application ran slower than the period, so the timing "
                            "cannot confirm the input rate; the configured period is used",
                            "declared", spec)
            return _row(key, label, exp, shown, VIOLATED,
                        f"inputs arrived slower than assumed ({observed:.4g} ms)",
                        "measured", spec)
        return _row(key, label, exp, shown, HOLDS, None, "measured", spec)

    if key == "max_start_temp_c":
        t = start_temp_c(result)
        if t is None:
            return _row(key, label, f"<= {want:g} C", None, NOT_VERIFIED,
                        "no temperature readings", spec=spec)
        ok = t <= want
        return _row(key, label, f"<= {want:g} C", t, HOLDS if ok else VIOLATED,
                    None if ok else "the board started hotter than assumed: let it cool down "
                    "(or pre-condition it) before measuring", "measured", spec)

    if key == "max_temp_c":
        tel = result.get("telemetry") or {}
        t = (result.get("soak") or {}).get("max_temp_c") or tel.get("max_temp_c")
        if not t:
            return _row(key, label, f"<= {want:g} C", None, NOT_VERIFIED,
                        "no temperature readings", spec=spec)
        ok = t <= want
        return _row(key, label, f"<= {want:g} C", t, HOLDS if ok else VIOLATED,
                    None if ok else "the run left the thermal envelope the contract is "
                    "claimed for", "measured", spec)

    if key == "max_background_cpu_pct":
        bg = (result.get("telemetry") or {}).get("background") or {}
        v = bg.get("cpu_pct")
        if v is None:
            return _row(key, label, f"<= {want:g}%", None, NOT_VERIFIED,
                        "the result has no background-load measurement (made with an older "
                        "EdgeLens, or imported)", spec=spec)
        ok = v <= want
        top = ", ".join(f"{t['name']} (pid {t['pid']}) {t['cores']:g} cores"
                        for t in (bg.get("top") or [])[:3])
        return _row(key, label, f"<= {want:g}%", v, HOLDS if ok else VIOLATED,
                    None if ok else (f"busiest: {top}; " if top else "") +
                    f"other processes used {v:g}% of the CPU "
                    f"{'(' + format(bg['cores'], 'g') + ' cores)' if bg.get('cores') is not None else ''}; stop them (desktop: sudo systemctl "
                    f"isolate multi-user.target)", "measured", spec)

    if key == "provider":
        got = _provider(result)
        if not got:
            return _row(key, label, want, None, NOT_VERIFIED,
                        "the result does not say which provider ran (not an ONNX run?)",
                        spec=spec)
        ok = got.lower() == want.lower()
        fallbacks = ((result.get("pipeline") or {}).get("config") or {}).get(
            "provider_fallbacks") or []
        return _row(key, label, want, got, HOLDS if ok else VIOLATED,
                    None if ok else f"ran on {got}" + (f" (fell back: {fallbacks})"
                                                       if fallbacks else ""),
                    "configuration", spec)

    if key == "model_sha256":
        got = (result.get("identity") or {}).get("model_sha256")
        if not got:
            return _row(key, label, want, None, NOT_VERIFIED,
                        "the result has no model hash (not run with --model)", spec=spec)
        a, b = got.lower(), want.lower()
        ok = a.startswith(b) or b.startswith(a)
        return _row(key, label, want, got, HOLDS if ok else VIOLATED,
                    None if ok else "a different model file was measured", "measured", spec)

    if key == "min_in_envelope_pct":
        dec = result.get("decision") or {}
        v = dec.get("in_envelope_pct")
        if dec.get("aggregate"):
            v = dec["aggregate"].get("in_envelope_pct_min")
        if v is None:
            return _row(key, label, f">= {want:g}%", None, NOT_VERIFIED,
                        "no envelope check in this run: fit one (edgelens envelope fit) and "
                        "name it in the stream spec", spec=spec)
        ok = v >= want
        return _row(key, label, f">= {want:g}%", v, HOLDS if ok else VIOLATED,
                    None if ok else "the model answered about inputs outside its training "
                    "data: `edgelens envelope check` names them", "measured", spec)

    if key == "stream_spec_sha256":
        got = ((result.get("stream") or {}).get("spec") or {}).get("sha256")
        if not got:
            return _row(key, label, want, None, NOT_VERIFIED,
                        "not an `edgelens replay` result", spec=spec)
        a, b = got.lower(), want.lower()
        ok = a.startswith(b) or b.startswith(a)
        return _row(key, label, want, got, HOLDS if ok else VIOLATED,
                    None if ok else "the pipeline was replayed with another spec "
                    "(scaler, window or post-processing changed)", "measured", spec)

    raise ContractError(f"Unknown assumption '{key}'.")      # pragma: no cover


def _evidence_rows(result, c):
    rows = []
    if c["min_duration_s"] is not None:
        d = run_duration_s(result)
        st = MET if (d is not None and d >= c["min_duration_s"] * 0.99) else NOT_MET
        rows.append({"condition": "min_duration", "label": "Run duration", "unit": "s",
                     "expected": f">= {c['min_duration_s']:g} s",
                     "measured": round(d, 1) if d is not None else None, "status": st,
                     "reason": None if st == MET else
                     ("unknown run duration" if d is None else
                      f"the run lasted {d:.1f} s")})
    if c["min_iterations"] is not None:
        n = result.get("iterations") or 0
        st = MET if n >= c["min_iterations"] else NOT_MET
        rows.append({"condition": "min_iterations", "label": "Iterations", "unit": None,
                     "expected": f">= {c['min_iterations']}", "measured": n, "status": st,
                     "reason": None if st == MET else f"the run had {n} iterations"})
    return rows


# ---------------------------------------------------------------- the verdict

def evaluate(result, contract, contract_sha256=None):
    """Check one result against a parsed contract. Returns a verdict document:
    the `edgelens validate` document plus `contract`, `assumptions` and
    `evidence_conditions`, with requirement IDs on every row."""
    c = contract
    reqs = {k: s["value"] for k, s in c["guarantees"].items()}
    out = validate(result, reqs)
    g_verdict = out["verdict"]

    # requirement IDs on the guarantee rows; deadline the contract specifies
    for row in out["checks"]:
        spec = c["guarantees"].get(row["requirement"]) or {}
        row["id"] = spec.get("id")
        row["note"] = spec.get("note")
        if row["requirement"] == "max_miss_ratio" and spec.get("deadline_ms") is not None:
            dl = (result.get("deadline") or {}).get("deadline_ms")
            # misses counted against another deadline prove nothing either way
            if dl is not None and abs(dl - spec["deadline_ms"]) > 1e-6:
                row["status"] = NOT_MEASURED
                row["measured"] = None
                row["reason"] = (f"the contract's deadline is {spec['deadline_ms']:g} ms; "
                                 f"the run used {dl:g} ms — rerun with --deadline-ms "
                                 f"{spec['deadline_ms']:g}")
    statuses = {r["status"] for r in out["checks"]}
    if result.get("mode") == "demo":
        g_verdict = INCONCLUSIVE
    elif FAIL in statuses:
        g_verdict = FAIL
    elif NOT_MEASURED in statuses:
        g_verdict = INCONCLUSIVE
    else:
        g_verdict = PASS

    assumptions = [check_assumption(result, k, s) for k, s in c["assumes"].items()]
    evidence = _evidence_rows(result, c)
    broken = [a for a in assumptions if a["status"] != HOLDS]
    thin = [e for e in evidence if e["status"] != MET]

    notes = [n for n in out["notes"] if not n.startswith("Some requirements")]
    if NOT_MEASURED in statuses:
        notes.append("Some guarantees could not be measured by this run; see each reason.")
    if broken or thin:
        verdict = INCONCLUSIVE
        for a in broken:
            notes.append(f"Assumption {a['assumption']}"
                         f"{' [' + a['id'] + ']' if a.get('id') else ''} "
                         f"{a['status'].replace('_', ' ')}: {a.get('reason') or ''}".rstrip(": "))
        for e in thin:
            notes.append(f"Too little evidence: {e['label'].lower()} {e['expected']} "
                         f"({e['reason']}).")
        if g_verdict == FAIL:
            notes.append("A guarantee FAILED, but outside the contract's assumptions: this "
                         "is not evidence against the contract. Fix the setup and rerun.")
        elif g_verdict == PASS:
            notes.append("Every guarantee was met, but not under the contract's "
                         "assumptions: that is not a PASS.")
    else:
        verdict = g_verdict

    out["verdict"] = verdict
    out["guarantees_verdict"] = g_verdict
    out["notes"] = notes
    out["assumptions"] = assumptions
    out["evidence_conditions"] = evidence
    out["contract"] = {"name": c["contract"], "version": c.get("version"),
                       "file": c.get("file"), "sha256": contract_sha256,
                       "requirements_doc": c.get("requirements_doc"),
                       "description": c.get("description")}
    if verdict != FAIL:
        out.pop("diagnosis", None)
        out.pop("limiting_stage", None)
    elif "diagnosis" not in out:
        from .validate.engine import failure_diagnosis
        out["limiting_stage"], out["diagnosis"] = failure_diagnosis(result, out["checks"])
    return out


# ---------------------------------------------------------------- evidence report

def _fmt(v, unit=None):
    """unit "" = a ratio shown as % (miss ratio, decision agreement); None = plain."""
    if v is None:
        return "—"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items())
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        if unit == "":
            return f"{v * 100:.3f}%"
        if unit == "%":
            return f"{v:g}%"
        return f"{v:g} {unit}" if unit else f"{v:g}"
    return f"{v} {unit}".strip() if unit else str(v)


def _md(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


ICON = {PASS: "✅ PASS", FAIL: "❌ FAIL", NOT_MEASURED: "⚠️ NOT MEASURED",
        HOLDS: "✅ holds", VIOLATED: "❌ violated", NOT_VERIFIED: "⚠️ not verified",
        MET: "✅ met", NOT_MET: "❌ not met", INCONCLUSIVE: "⚠️ INCONCLUSIVE"}


def evidence_markdown(verdict, result_path=None):
    """A Markdown evidence record: one row per requirement ID, the conditions
    it was measured under, and the run's identity. Paste-ready for a
    deliverable (D6.2) or a CI job summary."""
    c = verdict.get("contract") or {}
    run = verdict.get("run") or {}
    L = []
    L.append(f"# {ICON.get(verdict['verdict'], verdict['verdict'])} — {_md(c.get('name'))}"
             + (f" v{_md(c['version'])}" if c.get("version") else ""))
    L.append("")
    if c.get("description"):
        L.append(_md(c["description"]))
        L.append("")
    meta = [("Contract", f"{c.get('file') or '—'} (sha256 {str(c.get('sha256') or '')[:12]})"),
            ("Requirements", c.get("requirements_doc") or "—"),
            ("Result file", result_path or "—"),
            ("Board", run.get("board") or "—"),
            ("Power mode", run.get("power_mode") or "—"),
            ("Iterations", run.get("iterations")),
            ("Run id", run.get("run_id")),
            ("Environment id", run.get("environment_id")),
            ("Guarantees alone", verdict.get("guarantees_verdict"))]
    L.append("| | |")
    L.append("| --- | --- |")
    for k, v in meta:
        L.append(f"| {k} | {_md(v if v is not None else '—')} |")
    L.append("")
    if verdict.get("notes"):
        for n in verdict["notes"]:
            L.append(f"> {_md(n)}")
        L.append("")

    L.append("## Assumptions (conditions of the claim)")
    L.append("")
    if verdict.get("assumptions") or verdict.get("evidence_conditions"):
        L.append("| ID | Assumption | Expected | Measured | Status |")
        L.append("| --- | --- | --- | --- | --- |")
        for a in verdict.get("assumptions") or []:
            st = ICON[a["status"]] + (f" — {_md(a['reason'])}" if a.get("reason") else "")
            L.append(f"| {_md(a.get('id') or '')} | {_md(a['label'])} | {_md(_fmt(a['expected']))}"
                     f" | {_md(_fmt(a['measured'], a.get('unit')))} | {st} |")
        for e in verdict.get("evidence_conditions") or []:
            st = ICON[e["status"]] + (f" — {_md(e['reason'])}" if e.get("reason") else "")
            L.append(f"| | {_md(e['label'])} | {_md(e['expected'])} | {_md(_fmt(e['measured'], e.get('unit')))}"
                     f" | {st} |")
    else:
        L.append("_None stated: the guarantees are claimed unconditionally._")
    L.append("")

    L.append("## Guarantees")
    L.append("")
    if not verdict.get("checks"):
        L.append("_Not measured._")
    else:
        L.append("| ID | Requirement | Limit | Measured | Status |")
        L.append("| --- | --- | --- | --- | --- |")
    for r in verdict.get("checks") or []:
        op = "≤" if r["direction"] == "max" else "≥"
        st = ICON[r["status"]] + (f" — {_md(r['reason'])}" if r.get("reason") else "")
        L.append(f"| {_md(r.get('id') or '')} | {_md(r['label'])} | {op} {_md(_fmt(r['limit'], r['unit']))}"
                 f" | {_md(_fmt(r['measured'], r['unit']))} | {st} |")
    L.append("")
    issues = [i for i in verdict.get("integrity") or [] if i.get("severity") != "info"]
    if issues:
        L.append("## Measurement integrity")
        L.append("")
        for i in issues:
            L.append(f"- **{i['code']}** ({i['severity']}): {_md(i['detail'])}")
        L.append("")
    d = (verdict.get("diagnosis") or {}).get("primary")
    if d:
        L.append("## Why it failed")
        L.append("")
        ls = verdict.get("limiting_stage")
        if ls:
            L.append(f"- Limiting stage: **{_md(ls['stage'])}**")
        L.append(f"- {_md(d['type'])} ({_md(d['evidence_strength'])} evidence): {_md(d['detail'])}")
        L.append(f"- Next experiment: {_md(d['recommendation'])}")
        L.append("")
    L.append("_Generated by EdgeLens `validate --contract`. INCONCLUSIVE is never a PASS: "
             "it means the run did not measure the situation the contract is about._")
    return "\n".join(L) + "\n"


TEMPLATE = """\
# EdgeLens assumption contract. Check it with:  edgelens contract check THIS_FILE
# Validate a run with:                          edgelens validate run.json --contract THIS_FILE
contract: my-pipeline
version: "1.0"
description: Anomaly detection at 1 Hz on the target board
requirements_doc: D1.1            # where the requirement IDs come from

min_duration: 10m                 # less evidence than this -> INCONCLUSIVE
min_iterations: 300

# Conditions the guarantees are claimed under. Any of them violated (or not
# verifiable from the run) -> INCONCLUSIVE, never PASS.
assumes:
  board: Orin                     # substring of the board name
  power_mode: MAXN                # nvpmodel name (or id: 0)
  clocks_pinned: true             # sudo jetson_clocks
  input_period_ms: {value: 1000, tolerance_pct: 5}
  max_start_temp_c: 55            # let the board cool down first
  max_background_cpu_pct: 10      # nothing else running
  # provider: TensorrtExecutionProvider
  # model_sha256: 1a2b3c4d...

# What the system must do. Same keys as `edgelens validate` flags.
guarantees:
  p99_ms:          {limit: 3000, id: R-01}
  max_miss_ratio:  {limit: 0, deadline_ms: 3000, id: R-02}
  max_temp_c:      {limit: 85, id: R-03}
  # max_energy_per_period_mj: {limit: 15000, id: R-04}   # edgelens periodic
  # max_output_error: {limit: 0.01, id: R-05}            # edgelens sweep
  # max_latency_drift_pct: {limit: 10, id: R-06}         # edgelens soak
  # max_mem_growth_mb_per_h: {limit: 20, id: R-07}       # edgelens soak
  # max_throttled_windows: {limit: 0, id: R-08}          # edgelens soak
  # max_time_to_detect_s: {limit: 120, id: R-09}         # edgelens replay / detect
  # max_false_alarms_per_h: {limit: 0.5, id: R-10}       # edgelens replay / detect
  # max_missed_events: {limit: 0, id: R-11}              # edgelens replay / detect
"""
