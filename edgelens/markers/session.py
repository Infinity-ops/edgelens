"""
edgelens.markers.session
--------------------------
Turns marker spans (from any process, any language) into the standard
EdgeLens result document, so `diagnose`, `validate`, `compare` and `report`
work on a C++ application exactly as on a Python pipeline.

Iterations:
  * If the application marks whole iterations ("@iteration" spans), those
    give end-to-end latency, and every other span is assigned to the
    iteration (same process) that contains it.
  * Otherwise, per process, a new iteration starts whenever a span name
    repeats, and end-to-end latency runs from the first span's start to the
    last span's end — the same rule as observer mode.

GPU timing check: a span ended without GPU synchronisation measures only how
long the CPU took to *enqueue* asynchronous CUDA/TensorRT work. Inference
spans without the `g` flag are counted and reported, so a 0.3 ms
"inference" that really takes 9 ms on the GPU is flagged, not believed.
"""

import bisect

from ..core.identity import attach_identity
from ..core.pipeline import infer_role
from ..core.result import MAX_TRACE_EVENTS, build_result
from ..packs import get_pack
from .protocol import ITERATION


def _assign_explicit(spans):
    """Iterations from '@iteration' spans. A stage span belongs to the
    iteration that contains it on the same thread, else on the same process
    (work handed to a worker thread)."""
    iters = sorted((sp for sp in spans if sp["name"] == ITERATION), key=lambda s: s["start_ns"])
    index = {id(sp): i for i, sp in enumerate(iters)}
    e2e = [(sp["end_ns"] - sp["start_ns"]) / 1e6 for sp in iters]
    by_key = {}
    for sp in iters:
        by_key.setdefault((sp["pid"], sp["tid"]), []).append(sp)
        by_key.setdefault((sp["pid"], None), []).append(sp)
    starts = {k: [s["start_ns"] for s in v] for k, v in by_key.items()}
    # prefix maximum of end times: once it is below a span's end, no earlier
    # iteration can contain the span — keeps the lookup from going quadratic.
    reach = {}
    for k, v in by_key.items():
        m, acc = -1, []
        for it in v:
            m = max(m, it["end_ns"])
            acc.append(m)
        reach[k] = acc

    def find(key, sp):
        lst = by_key.get(key)
        if not lst:
            return -1
        k = bisect.bisect_right(starts[key], sp["start_ns"]) - 1
        pm = reach[key]
        while k >= 0 and pm[k] >= sp["end_ns"]:
            if lst[k]["end_ns"] >= sp["end_ns"]:
                return index[id(lst[k])]
            k -= 1
        return -1

    def iteration_of(sp):
        i = find((sp["pid"], sp["tid"]), sp)
        return i if i >= 0 else find((sp["pid"], None), sp)
    return e2e, iteration_of, 1, [sp["start_ns"] for sp in iters]


def _assign_auto(spans):
    """Iterations by name repetition, per (process, thread)."""
    by_stream = {}
    for sp in sorted(spans, key=lambda s: s["start_ns"]):
        by_stream.setdefault((sp["pid"], sp["tid"]), []).append(sp)
    bounds = []          # (start, end, members) per iteration, all streams
    it_of = {}
    for lst in by_stream.values():
        seen, cur = set(), None
        for sp in lst:
            if cur is None or sp["name"] in seen:
                if cur is not None:
                    bounds.append(cur)
                cur = [sp["start_ns"], sp["end_ns"], []]
                seen = set()
            seen.add(sp["name"])
            cur[1] = max(cur[1], sp["end_ns"])
            cur[2].append(sp)
        if cur is not None:
            bounds.append(cur)
    bounds.sort(key=lambda b: b[0])
    e2e = []
    for i, (s, e, members) in enumerate(bounds):
        e2e.append((e - s) / 1e6)
        for sp in members:
            it_of[id(sp)] = i
    return e2e, (lambda sp: it_of.get(id(sp), -1)), len(by_stream), [b[0] for b in bounds]


def gpu_sync_summary(spans, roles):
    inference = [sp for sp in spans if roles.get(sp["name"]) == "inference"]
    unsynced = [sp for sp in inference if "g" not in sp["flags"]]
    return {"inference_spans": len(inference),
            "gpu_synced_spans": sum(1 for sp in spans if "g" in sp["flags"]),
            "unsynced_inference_spans": len(unsynced),
            "unsynced_inference_stages": sorted({sp["name"] for sp in unsynced})}


def spans_to_result(spans, marker_stats, *, name="markers", pack="custom", deadline_ms=None,
                    period_ms=None, telemetry_summary=None, telemetry_series=None,
                    origin_ns=None, mode="markers", pipeline_source=None, source="socket",
                    attach_ids=True, extra=None, idle_power_w=None):
    pack_obj = get_pack(pack)
    cfg = pack_obj.configure({})
    if deadline_ms is None:
        deadline_ms = period_ms if period_ms is not None else pack_obj.default_deadline_ms(cfg)
    stage_spans = [sp for sp in spans if sp["name"] != ITERATION]
    has_explicit = any(sp["name"] == ITERATION for sp in spans)
    if has_explicit:
        e2e, iteration_of, streams, iter_starts = _assign_explicit(spans)
    else:
        e2e, iteration_of, streams, iter_starts = _assign_auto(stage_spans)

    order, samples = [], {}
    for sp in sorted(stage_spans, key=lambda s: s["start_ns"]):
        if sp["name"] not in samples:
            order.append(sp["name"])
            samples[sp["name"]] = []
        samples[sp["name"]].append((sp["end_ns"] - sp["start_ns"]) / 1e6)
    roles = {n: pack_obj.role_for(n) or infer_role(n) for n in order}

    if origin_ns is None:
        origin_ns = min((sp["start_ns"] for sp in spans), default=0)
    events, truncated = [], False
    for sp in sorted(spans, key=lambda s: s["start_ns"]):
        if len(events) >= MAX_TRACE_EVENTS:
            truncated = True
            break
        events.append({"iter": iteration_of(sp) if sp["name"] != ITERATION else None,
                       "stage": sp["name"], "start_ns": sp["start_ns"] - origin_ns,
                       "dur_ns": sp["end_ns"] - sp["start_ns"], "thread": sp["tid"],
                       "pid": sp["pid"], "flags": sp["flags"]})

    desc = {"name": name, "pack": pack_obj.name, "config": cfg,
            "stages": [{"name": n, "role": roles[n]} for n in order]}
    markers_block = dict(marker_stats or {})
    markers_block["source"] = source
    markers_block["iterations_from"] = "@iteration spans" if has_explicit else \
        "name repetition (add @iteration spans for exact end-to-end latency)"
    markers_block["gpu_timing"] = gpu_sync_summary(stage_spans, roles)
    markers_block["processes"] = sorted({sp["pid"] for sp in spans})
    # Several threads/processes and no @iteration spans: which spans form one
    # end-to-end iteration is a guess (integrity: ITERATIONS_AMBIGUOUS).
    markers_block["iteration_streams"] = streams
    first = min((sp["start_ns"] for sp in spans), default=None)
    last = max((sp["end_ns"] for sp in spans), default=None)
    # Observed rate: n iterations whose starts span (n - 1) intervals.
    window_ms = None
    if len(iter_starts) > 1 and max(iter_starts) > min(iter_starts):
        n = len(iter_starts)
        window_ms = (max(iter_starts) - min(iter_starts)) / 1e6 * n / (n - 1)

    result = build_result(
        pipeline_desc=desc, stage_samples_ms=samples, e2e_ms=e2e, mode=mode,
        pipeline_source=pipeline_source or f"markers:{name}",
        telemetry_summary=telemetry_summary, telemetry_series=telemetry_series,
        events=events, events_truncated=truncated, deadline_ms=deadline_ms,
        period_ms=period_ms, paced=False, pack=pack_obj, trace_origin_ns=origin_ns,
        idle_power_w=idle_power_w, extra=dict(extra or {}, markers=markers_block),
        throughput_window_ms=window_ms,
        energy_window_ns=(first, last) if spans else None,
    )
    if attach_ids:
        attach_identity(result)
    return result
