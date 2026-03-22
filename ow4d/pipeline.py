from pathlib import Path

from ow4d.config import load_config
from ow4d.io import ensure_dir, save_json, save_csv, load_json
from ow4d.prompts import parse_prompt_groups
from ow4d.inputs import inspect_input, sample_frame_ids, export_sampled_frames
from ow4d.adapters.registry import run_adapter
from ow4d.fusion import smooth_observations
from ow4d.stabilization import (
    canonicalize_rows,
    canonicalize_anchor_rows,
    stabilize_rows,
    apply_windowed_anchor_fusion,
)
from ow4d.scene_graph import build_scene_graph
from ow4d.track_state import build_track_state
from ow4d.lifting import build_object_state_4d
from ow4d.lifters import lift_object_state_4d
from ow4d.scene_graph_4d import build_scene_graph_4d
from ow4d.dynamics import build_scene_dynamics
from ow4d.object_graph import build_dynamic_object_graph
from ow4d.forecast_prep import build_forecast_samples
from ow4d.future_graph_prep import build_future_graph_samples_from_scene_graph_4d
from ow4d.forecast_3d import (
    build_forecast_samples_3d,
    build_constant_position_baseline_3d,
    build_constant_velocity_baseline_3d,
    build_forecast_baseline_compare_3d,
)
from ow4d.future_graph_baseline import (
    build_constant_position_future_graph_baseline,
    build_hybrid_future_graph_baseline,
    build_future_graph_baseline_compare,
)
from ow4d.forecast_baseline import (
    build_constant_position_baseline,
    build_constant_velocity_baseline,
    build_hybrid_velocity_position_baseline,
    build_forecast_baseline_compare,
)
from ow4d.render import render_keyframes
from ow4d.tracks import summarize_tracks
from ow4d.manifest import make_manifest

def run_core(input_path, prompt_text, config_path=None, out_dir="outputs/run", adapter=None, render=True, stride=None, max_frames=None, pred_path=None, anchor_path=None, lift_mode=None, depth_dir=None, intrinsics_path=None, temporal_stabilization=None, anchor_fusion=None, stabilization_alpha=None, stabilization_max_gap=None, stabilization_max_center_step=None, anchor_fusion_window=None, anchor_fusion_weight=None):
    cfg = load_config(config_path)
    if adapter:
        cfg["adapter"] = adapter
    if lift_mode is not None:
        cfg["lift_mode"] = lift_mode
    if depth_dir is not None:
        cfg["depth_dir"] = depth_dir
    if intrinsics_path is not None:
        cfg["intrinsics_path"] = intrinsics_path
    if temporal_stabilization is not None:
        cfg["temporal_stabilization"] = temporal_stabilization
    if anchor_fusion is not None:
        cfg["anchor_fusion"] = anchor_fusion
    if stabilization_alpha is not None:
        cfg["stabilization_alpha"] = stabilization_alpha
    if stabilization_max_gap is not None:
        cfg["stabilization_max_gap"] = stabilization_max_gap
    if stabilization_max_center_step is not None:
        cfg["stabilization_max_center_step"] = stabilization_max_center_step
    if anchor_fusion_window is not None:
        cfg["anchor_fusion_window"] = anchor_fusion_window
    if anchor_fusion_weight is not None:
        cfg["anchor_fusion_weight"] = anchor_fusion_weight

    prompt_groups = parse_prompt_groups(prompt_text)
    if not prompt_groups:
        raise RuntimeError("no prompts")

    out_dir = Path(out_dir)
    key_dir = out_dir / "keyframes"
    ensure_dir(out_dir)
    ensure_dir(key_dir)

    info = inspect_input(input_path)

    if stride is None:
        if info["input_type"] == "sequence":
            stride = cfg["sequence_stride"]
        else:
            stride = cfg["key_frame_stride"]

    if max_frames is None:
        max_frames = cfg["max_keyframes"]

    frame_ids = sample_frame_ids(
        info["frame_count"],
        stride=stride,
        max_frames=max_frames
    )
    frames = export_sampled_frames(input_path, key_dir, frame_ids)

    observations = run_adapter(
        cfg["adapter"],
        frames,
        prompt_groups,
        info["width"],
        info["height"],
        pred_path=pred_path,
    )
    observations, canon_meta = canonicalize_rows(observations, prompt_groups)
    raw_observations = [dict(row) for row in observations]

    anchors = []
    anchor_canon_meta = {"rows": 0, "changed": 0}
    if anchor_path:
        anchors = load_json(anchor_path)
        anchors, anchor_canon_meta = canonicalize_anchor_rows(anchors, prompt_groups)

    stab_meta = {
        "enabled": False,
        "input_rows": len(observations),
        "output_rows": len(observations),
        "gap_filled_rows": 0,
        "track_groups": 0,
    }
    if cfg.get("temporal_stabilization", False):
        observations, stab_meta = stabilize_rows(
            observations,
            alpha=cfg["stabilization_alpha"],
            max_gap=cfg["stabilization_max_gap"],
            min_score=cfg["stabilization_min_score"],
            max_center_step=cfg["stabilization_max_center_step"],
        )
        stab_meta["enabled"] = True

    stabilized_observations = [dict(row) for row in observations]

    anchor_meta = {
        "enabled": False,
        "input_rows": len(stabilized_observations),
        "anchor_rows": len(anchors),
        "fused_rows": 0,
    }
    canonical_observations = [dict(row) for row in stabilized_observations]
    if anchors and cfg.get("anchor_fusion", False):
        canonical_observations, anchor_meta = apply_windowed_anchor_fusion(
            canonical_observations,
            anchors,
            window=cfg["anchor_fusion_window"],
            base_weight=cfg["anchor_fusion_weight"],
            min_iou=cfg["anchor_fusion_min_iou"],
            max_center_frac=cfg["anchor_fusion_max_center_frac"],
        )
        anchor_meta["enabled"] = True

    observations = [dict(row) for row in canonical_observations]
    track_state = build_track_state(canonical_observations)
    object_state_4d = build_object_state_4d(
        track_state,
        info,
    )
    lifted_object_state_4d = lift_object_state_4d(
        object_state_4d,
        info,
        lift_mode=cfg.get("lift_mode", "image_plane"),
        depth_dir=cfg.get("depth_dir"),
        intrinsics_path=cfg.get("intrinsics_path"),
    )
    scene_graph_4d = build_scene_graph_4d(
        lifted_object_state_4d,
        near_thresh=cfg["near_thresh"],
    )
    scene_dynamics = build_scene_dynamics(
        track_state,
        near_thresh=cfg["near_thresh"],
    )
    dynamic_object_graph = build_dynamic_object_graph(
        track_state,
        scene_dynamics,
    )
    forecast_samples = build_forecast_samples(
        track_state,
        obs_len=4,
        pred_len=2,
        min_speed=0.0,
    )
    future_graph_samples = build_future_graph_samples_from_scene_graph_4d(
        scene_graph_4d,
        obs_len=4,
        pred_len=2,
    )
    forecast_samples_3d = build_forecast_samples_3d(
        lifted_object_state_4d,
        obs_len=4,
        pred_len=2,
    )
    forecast_baseline_cp_3d = build_constant_position_baseline_3d(
        forecast_samples_3d,
    )
    forecast_baseline_cv_3d = build_constant_velocity_baseline_3d(
        forecast_samples_3d,
    )
    forecast_baseline_compare_3d = build_forecast_baseline_compare_3d(
        forecast_baseline_cp_3d,
        forecast_baseline_cv_3d,
    )
    future_graph_baseline_cp = build_constant_position_future_graph_baseline(
        future_graph_samples,
        near_thresh=cfg["near_thresh"],
    )
    future_graph_baseline_hybrid = build_hybrid_future_graph_baseline(
        future_graph_samples,
        near_thresh=cfg["near_thresh"],
    )
    future_graph_baseline_compare = build_future_graph_baseline_compare(
        future_graph_baseline_cp,
        future_graph_baseline_hybrid,
    )
    forecast_baseline_cp = build_constant_position_baseline(
        forecast_samples,
    )
    forecast_baseline_cv = build_constant_velocity_baseline(
        forecast_samples,
    )
    forecast_baseline_hybrid = build_hybrid_velocity_position_baseline(
        forecast_samples,
    )
    forecast_baseline_compare = build_forecast_baseline_compare(
        forecast_baseline_cp,
        forecast_baseline_cv,
        forecast_baseline_hybrid,
    )

    if cfg["temporal_smoothing"]:
        observations = smooth_observations(
            observations,
            radius=cfg["smooth_radius"]
        )

    graph = build_scene_graph(observations, near_thresh=cfg["near_thresh"])
    track_summary = summarize_tracks(observations)

    save_json(out_dir / "video_info.json", info)
    save_json(out_dir / "prompts.json", prompt_groups)
    save_json(out_dir / "frames.json", frames)
    if cfg.get("write_raw_tracks", True):
        save_json(out_dir / "tracks_raw.json", raw_observations)
    save_json(out_dir / "tracks_stabilized.json", stabilized_observations)
    save_json(out_dir / "tracks_canonical.json", canonical_observations)
    save_json(out_dir / "track_state.json", track_state)
    save_json(out_dir / "object_state_4d.json", object_state_4d)
    save_json(out_dir / "lifted_object_state_4d.json", lifted_object_state_4d)
    save_json(out_dir / "lift_depth_match.json", lifted_object_state_4d.get("depth_match", {}))
    save_json(out_dir / "scene_graph_4d.json", scene_graph_4d)
    save_json(out_dir / "scene_dynamics.json", scene_dynamics)
    save_json(out_dir / "dynamic_object_graph.json", dynamic_object_graph)
    save_json(out_dir / "forecast_samples.json", forecast_samples)
    save_json(out_dir / "forecast_samples_3d.json", forecast_samples_3d)
    save_json(out_dir / "forecast_baseline_cp_3d.json", forecast_baseline_cp_3d)
    save_json(out_dir / "forecast_baseline_cv_3d.json", forecast_baseline_cv_3d)
    save_json(out_dir / "forecast_baseline_compare_3d.json", forecast_baseline_compare_3d)
    save_json(out_dir / "future_graph_samples.json", future_graph_samples)
    save_json(out_dir / "future_graph_baseline_cp.json", future_graph_baseline_cp)
    save_json(out_dir / "future_graph_baseline_hybrid.json", future_graph_baseline_hybrid)
    save_json(out_dir / "future_graph_baseline_compare.json", future_graph_baseline_compare)
    save_json(out_dir / "forecast_baseline_cp.json", forecast_baseline_cp)
    save_json(out_dir / "forecast_baseline_cv.json", forecast_baseline_cv)
    save_json(out_dir / "forecast_baseline_hybrid.json", forecast_baseline_hybrid)
    save_json(out_dir / "forecast_baseline_compare.json", forecast_baseline_compare)
    save_json(out_dir / "tracks.json", observations)
    save_json(out_dir / "scene_graph.json", graph)
    save_csv(out_dir / "track_summary.csv", track_summary)

    preview_path = None
    if render:
        preview_path = render_keyframes(
            frames,
            observations,
            out_dir / "preview_keyframes.mp4",
            fps=6
        )

    manifest = make_manifest(
        video_path=input_path,
        prompt_text=prompt_text,
        prompt_groups=prompt_groups,
        cfg=cfg,
        info=info,
        frames=frames,
        observations=observations,
        out_dir=out_dir,
    )
    manifest["effective_stride"] = stride
    manifest["effective_max_frames"] = max_frames
    manifest["pred_path"] = pred_path
    manifest["anchor_path"] = anchor_path
    manifest["label_canonicalization"] = canon_meta
    manifest["anchor_label_canonicalization"] = anchor_canon_meta
    manifest["temporal_stabilization"] = stab_meta
    manifest["anchor_fusion"] = anchor_meta
    manifest["lifting"] = {
        "lift_mode": cfg.get("lift_mode", "image_plane"),
        "depth_dir": cfg.get("depth_dir"),
        "intrinsics_path": cfg.get("intrinsics_path"),
        "representation": lifted_object_state_4d.get("representation"),
        "coord_frame": lifted_object_state_4d.get("coord_frame"),
        "valid_3d": lifted_object_state_4d.get("valid_3d"),
        "lift_meta": lifted_object_state_4d.get("lift_meta", {}),
    }
    manifest["track_artifacts"] = {
        "raw": "tracks_raw.json" if cfg.get("write_raw_tracks", True) else None,
        "stabilized": "tracks_stabilized.json",
        "canonical": "tracks_canonical.json",
        "canonical_source": ["tracks_stabilized.json", "anchors.json"] if anchors and cfg.get("anchor_fusion", False) else "tracks_stabilized.json",
        "state": "track_state.json",
        "state_source": "tracks_canonical.json",
        "object_state_4d": "object_state_4d.json",
        "object_state_4d_source": "track_state.json",
        "lifted_object_state_4d": "lifted_object_state_4d.json",
        "lifted_object_state_4d_source": "object_state_4d.json",
        "lift_depth_match": "lift_depth_match.json",
        "lift_depth_match_source": ["object_state_4d.json", "depth_dir", "intrinsics_path"],
        "scene_graph_4d": "scene_graph_4d.json",
        "scene_graph_4d_source": "lifted_object_state_4d.json",
        "dynamics": "scene_dynamics.json",
        "dynamics_source": "track_state.json",
        "dynamic_object_graph": "dynamic_object_graph.json",
        "dynamic_object_graph_source": "track_state.json",
        "forecast_samples": "forecast_samples.json",
        "forecast_source": "track_state.json",
        "forecast_samples_3d": "forecast_samples_3d.json",
        "forecast_source_3d": "lifted_object_state_4d.json",
        "forecast_baseline_cp_3d": "forecast_baseline_cp_3d.json",
        "forecast_baseline_cp_3d_source": "forecast_samples_3d.json",
        "forecast_baseline_cv_3d": "forecast_baseline_cv_3d.json",
        "forecast_baseline_cv_3d_source": "forecast_samples_3d.json",
        "forecast_baseline_compare_3d": "forecast_baseline_compare_3d.json",
        "forecast_baseline_compare_3d_source": ["forecast_baseline_cp_3d.json", "forecast_baseline_cv_3d.json"],
        "future_graph_samples": "future_graph_samples.json",
        "future_graph_source": "scene_graph_4d.json",
        "future_graph_baseline_cp": "future_graph_baseline_cp.json",
        "future_graph_baseline_cp_source": "future_graph_samples.json",
        "future_graph_baseline_hybrid": "future_graph_baseline_hybrid.json",
        "future_graph_baseline_hybrid_source": "future_graph_samples.json",
        "future_graph_baseline_compare": "future_graph_baseline_compare.json",
        "future_graph_baseline_compare_source": ["future_graph_baseline_cp.json", "future_graph_baseline_hybrid.json"],
        "forecast_baseline_cp": "forecast_baseline_cp.json",
        "forecast_baseline_cp_source": "forecast_samples.json",
        "forecast_baseline_cv": "forecast_baseline_cv.json",
        "forecast_baseline_cv_source": "forecast_samples.json",
        "forecast_baseline_hybrid": "forecast_baseline_hybrid.json",
        "forecast_baseline_hybrid_source": "forecast_samples.json",
        "forecast_baseline_compare": "forecast_baseline_compare.json",
        "forecast_baseline_compare_source": ["forecast_baseline_cp.json", "forecast_baseline_cv.json", "forecast_baseline_hybrid.json"],
        "final": "tracks.json",
        "scene_graph_source": "tracks.json",
    }
    manifest["effective_stabilization_config"] = {
        "temporal_stabilization": cfg.get("temporal_stabilization"),
        "stabilization_alpha": cfg.get("stabilization_alpha"),
        "stabilization_max_gap": cfg.get("stabilization_max_gap"),
        "stabilization_min_score": cfg.get("stabilization_min_score"),
        "stabilization_max_center_step": cfg.get("stabilization_max_center_step"),
        "anchor_fusion": cfg.get("anchor_fusion"),
        "anchor_fusion_window": cfg.get("anchor_fusion_window"),
        "anchor_fusion_weight": cfg.get("anchor_fusion_weight"),
        "anchor_fusion_min_iou": cfg.get("anchor_fusion_min_iou"),
        "anchor_fusion_max_center_frac": cfg.get("anchor_fusion_max_center_frac"),
    }

    if preview_path:
        manifest["preview_path"] = preview_path

    save_json(out_dir / "manifest.json", manifest)

    return {
        "input_type": info["input_type"],
        "video_info": info,
        "prompt_groups": prompt_groups,
        "frame_count": len(frames),
        "obs_count": len(observations),
        "track_count": len(track_summary),
        "preview_path": preview_path,
        "effective_stride": stride,
        "effective_max_frames": max_frames,
        "pred_path": pred_path,
        "anchor_path": anchor_path,
        "label_canonicalization": canon_meta,
        "anchor_label_canonicalization": anchor_canon_meta,
        "temporal_stabilization": stab_meta,
        "anchor_fusion": anchor_meta,
        "out_dir": str(out_dir)
    }
