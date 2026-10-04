"""
Timeseries pack: sensor / signal / audio / IMU / vibration / CAN pipelines.

Config (all in samples, per channel):
    sample_rate_hz   sensor sampling rate
    window           samples per analysis window
    hop              samples between consecutive windows (default = window,
                     i.e. no overlap)

Derived:
    hop_period_ms        new data arrives every hop / sample_rate seconds.
                         This is the real-time period AND the default deadline:
                         a window must finish before the next one is ready,
                         or a backlog builds up.
    window_duration_ms   signal time covered by one window
    overlap_pct          share of each window re-processed by the next one

Headline metric: real-time factor (RTF) = processing time / hop period.
RTF < 1 keeps up with the sensor; RTF >= 1 falls further behind forever.
"""

import math
from pathlib import Path

import numpy as np

from .base import Pack


class TimeseriesPack(Pack):
    name = "timeseries"
    description = ("Sensor/signal/audio pipelines (acquire -> filter -> features -> infer "
                   "-> decide) with windowing and real-time factor.")
    stage_roles = {
        "acquire": "input",
        "filter": "preprocess",
        "resample": "preprocess",
        "window": "preprocess",
        "fft": "preprocess",
        "feature_extraction": "preprocess",
        "inference": "inference",
        "decision": "decision",
    }

    def configure(self, config):
        sr, win = config.get("sample_rate_hz"), config.get("window")
        hop = config.get("hop", win)
        if sr is None and win is None:
            return config
        if not sr or sr <= 0:
            raise ValueError("timeseries pack: sample_rate_hz must be > 0")
        if not win or win <= 0:
            raise ValueError("timeseries pack: window (samples) must be > 0")
        if not hop or hop <= 0 or hop > win:
            raise ValueError("timeseries pack: hop must be in 1..window")
        config["hop"] = hop
        config["hop_period_ms"] = round(1000.0 * hop / sr, 4)
        config["window_duration_ms"] = round(1000.0 * win / sr, 4)
        config["overlap_pct"] = round(100.0 * (1 - hop / win), 2)
        return config

    def default_deadline_ms(self, config):
        return config.get("hop_period_ms")

    def default_period_ms(self, config):
        return config.get("hop_period_ms")

    def template(self):
        return [("acquire", "input"), ("filter", "preprocess"),
                ("feature_extraction", "preprocess"), ("inference", "inference"),
                ("decision", "decision")]

    def metrics(self, result):
        cfg = (result.get("pipeline") or {}).get("config") or {}
        req = result.get("requirements") or {}
        # The period inputs actually arrived at: an explicit --period-ms wins
        # over the hop period derived from sample_rate/hop.
        period = req.get("period_ms") or cfg.get("hop_period_ms")
        svc = result.get("service_latency") or result.get("latency") or {}
        if not period or not svc.get("n"):
            return {}

        def rtf(v):
            return None if v is None else round(v / period, 4)

        mean = svc["mean"]
        sr = cfg.get("sample_rate_hz")
        return {
            "period_ms": period,
            "hop_period_ms": cfg.get("hop_period_ms"),
            "window_duration_ms": cfg.get("window_duration_ms"),
            "overlap_pct": cfg.get("overlap_pct"),
            # RTF = SERVICE time / period. Never response time: in paced mode
            # that includes queueing and would overstate the factor.
            "real_time_factor": {"mean": rtf(mean), "p99": rtf(svc.get("p99")),
                                 "max": rtf(svc.get("max"))},
            "keeps_up_on_average": mean < period,
            "headroom_pct": round(100.0 * (1 - mean / period), 2),
            # Highest sample rate sustainable on average with this window/hop.
            "max_sustainable_sample_rate_hz": (round(sr * cfg["hop_period_ms"] / mean, 1)
                                               if (sr and cfg.get("hop_period_ms") and mean > 0)
                                               else None),
        }

    def findings(self, result):
        m = (result.get("pack_metrics") or {})
        if not m.get("period_ms"):
            return []
        period = m["period_ms"]
        r = m["real_time_factor"]
        backlog = result.get("backlog") or {}
        out = []
        if r["mean"] is not None and r["mean"] >= 1.0:
            out.append({
                "type": "CANNOT_KEEP_UP",
                "rank_score": 0.95,
                "detail": f"Mean processing time is {r['mean']:.2f}x the hop period "
                          f"({period} ms): new windows arrive faster than they are "
                          f"processed, so the backlog grows without bound and samples "
                          f"will eventually be dropped.",
                "recommendation": "Increase the hop (less overlap), shorten the window, "
                                  "move filtering/FFT to the GPU, or use a lighter model. "
                                  f"At this speed the pipeline sustains about "
                                  f"{m.get('max_sustainable_sample_rate_hz')} Hz.",
                "evidence": {"real_time_factor_mean": r["mean"], "period_ms": period,
                             "max_sustainable_sample_rate_hz": m.get("max_sustainable_sample_rate_hz")},
            })
        elif r["max"] is not None and r["max"] >= 1.0:
            out.append({
                "type": "TAIL_OVERRUNS_HOP",
                "rank_score": 0.7,
                "detail": f"On average the pipeline keeps up (RTF {r['mean']:.2f}), but "
                          f"the slowest windows take {r['max']:.2f}x the input period, so "
                          f"short backlogs form behind them.",
                "recommendation": "Look for periodic stalls (GC, thermal, other processes) "
                                  "in the trace; pin threads or raise priority; buffer at "
                                  "least max_backlog_items windows.",
                "evidence": {"real_time_factor_mean": r["mean"],
                             "real_time_factor_max": r["max"],
                             "max_backlog_items": backlog.get("max_backlog_items"),
                             "backlog_method": backlog.get("method")},
            })
        overlap = m.get("overlap_pct") or 0
        if overlap >= 75 and r["mean"] is not None and r["mean"] >= 0.5:
            out.append({
                "type": "HIGH_OVERLAP_COST",
                "rank_score": 0.55,
                "detail": f"Windows overlap by {overlap:.0f}%, so most of each window "
                          f"is processed again by the next one, and the pipeline already "
                          f"uses {r['mean']*100:.0f}% of its time budget.",
                "recommendation": "Check whether the model really needs this overlap; a "
                                  "larger hop cuts compute and energy proportionally.",
                "evidence": {"overlap_pct": overlap, "real_time_factor_mean": r["mean"]},
            })
        return out


class WindowSource:
    """Replays a recorded signal as a stream of windows.

        src = WindowSource("vibration.npy", window=1024, hop=512)   # or an array
        pipe.run(source=src, ...)

    The array's LAST axis is time (shape [T] or [channels, T]). Windows wrap
    around at the end so any iteration count works. Fetching a window from
    the source is not timed — time your own acquisition as the first stage.
    """

    def __init__(self, data, window, hop=None):
        if isinstance(data, (str, Path)):
            data = np.load(str(data))
        self.data = np.asarray(data)
        self.window = int(window)
        self.hop = int(hop or window)
        length = self.data.shape[-1]
        if length < self.window:
            raise ValueError(f"Signal has {length} samples, shorter than one window ({self.window}).")
        self.n_windows = 1 + (length - self.window) // self.hop
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        start = (self._i % self.n_windows) * self.hop
        self._i += 1
        return self.data[..., start:start + self.window]

    @property
    def windows_served(self):
        return self._i

    @staticmethod
    def synthetic(sample_rate_hz, seconds, channels=1, seed=0):
        """A deterministic test signal (two tones + noise), shape [channels, T]."""
        rng = np.random.default_rng(seed)
        t = np.arange(int(math.ceil(sample_rate_hz * seconds))) / sample_rate_hz
        base = np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)
        sig = np.stack([base + 0.1 * rng.standard_normal(t.size) for _ in range(channels)])
        return sig.astype(np.float32)
