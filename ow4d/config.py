from pathlib import Path
import yaml

DEFAULTS = {
    "key_frame_stride": 8,
    "sequence_stride": 1,
    "max_keyframes": 120,
    "temporal_smoothing": False,
    "smooth_radius": 2,
    "temporal_stabilization": True,
    "stabilization_alpha": 0.6,
    "stabilization_max_gap": 3,
    "stabilization_min_score": 0.0,
    "stabilization_max_center_step": 40.0,
    "anchor_fusion": True,
    "anchor_fusion_window": 2,
    "anchor_fusion_weight": 0.35,
    "anchor_fusion_min_iou": 0.25,
    "anchor_fusion_max_center_frac": 0.35,
    "write_raw_tracks": True,
    "lift_mode": "image_plane",
    "depth_dir": None,
    "intrinsics_path": None,
    "near_thresh": 140,
    "adapter": "mock",
}

def load_config(path=None):
    cfg = dict(DEFAULTS)
    if path:
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
            cfg.update(user_cfg)
    return cfg
