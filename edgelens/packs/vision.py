"""Vision pack: camera / video / image pipelines.

The classic six EdgeLens stages are this pack's template. Optional config:
    target_fps  -> per-frame deadline and period of 1000/target_fps ms
"""

from .base import Pack

VISION_STAGE_NAMES = ("capture", "preprocess", "h2d_copy", "inference", "d2h_copy", "postprocess")


class VisionPack(Pack):
    name = "vision"
    description = "Camera/video/image pipelines (capture -> preprocess -> infer -> postprocess)."
    stage_roles = {
        "capture": "input",
        "decode": "preprocess",
        "preprocess": "preprocess",
        "h2d_copy": "transfer",
        "inference": "inference",
        "d2h_copy": "transfer",
        "postprocess": "postprocess",
        "nms": "postprocess",
    }

    def configure(self, config):
        fps = config.get("target_fps")
        if fps is not None:
            if fps <= 0:
                raise ValueError("vision pack: target_fps must be > 0")
            config["frame_budget_ms"] = round(1000.0 / fps, 3)
        return config

    def default_deadline_ms(self, config):
        return config.get("frame_budget_ms")

    def default_period_ms(self, config):
        return config.get("frame_budget_ms")

    def template(self):
        return [(n, self.stage_roles[n]) for n in VISION_STAGE_NAMES]

    def metrics(self, result):
        cfg = (result.get("pipeline") or {}).get("config") or {}
        out = {"fps": result.get("fps")}
        if cfg.get("target_fps"):
            out["target_fps"] = cfg["target_fps"]
            out["meets_target_fps_on_average"] = (result.get("fps") or 0) >= cfg["target_fps"]
        return out

    def findings(self, result):
        cfg = (result.get("pipeline") or {}).get("config") or {}
        target = cfg.get("target_fps")
        fps = result.get("fps") or 0
        if not target or fps >= target:
            return []
        shortfall = 100.0 * (1 - fps / target)
        return [{
            "type": "BELOW_TARGET_FPS",
            "rank_score": round(min(0.9, 0.5 + shortfall / 100), 2),
            "detail": f"Average throughput is {fps:.1f} FPS against a {target} FPS target "
                      f"({shortfall:.0f}% short).",
            "recommendation": "See the stage breakdown for where the frame budget goes; "
                              "overlap capture/preprocess of frame N+1 with inference of "
                              "frame N, or reduce input resolution.",
            "evidence": {"fps": fps, "target_fps": target,
                         "frame_budget_ms": cfg.get("frame_budget_ms")},
        }]
