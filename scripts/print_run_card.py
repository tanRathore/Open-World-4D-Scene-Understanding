import argparse
import json
from pathlib import Path


def load_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def first_valid_3d(lifted):
    if not lifted:
        return None
    for obj in lifted.get("objects", []):
        for s in obj.get("states", []):
            if s.get("valid_3d"):
                return {
                    "track_id": obj.get("track_id"),
                    "label": obj.get("label"),
                    "frame_idx": s.get("frame_idx"),
                    "depth": s.get("depth"),
                    "world_xyz": s.get("world_xyz"),
                    "world_size_wh": s.get("world_size_wh"),
                    "bbox_3d": s.get("bbox_3d"),
                    "depth_stats": s.get("depth_stats"),
                }
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)

    manifest = load_json(run_dir / "manifest.json") or {}
    lifted = load_json(run_dir / "lifted_object_state_4d.json") or {}
    scene_graph_4d = load_json(run_dir / "scene_graph_4d.json") or {}
    fc3d = load_json(run_dir / "forecast_baseline_compare_3d.json") or {}
    fc2d = load_json(run_dir / "forecast_baseline_compare.json") or {}

    print("run_dir", str(run_dir))
    print("controller_path", [
        "preds.json",
        "label_canonicalization",
        "temporal_stabilization",
        "anchor_fusion",
        "track_state",
        "object_state_4d",
        "lifted_object_state_4d",
        "scene_graph_4d",
    ])
    print("temporal", manifest.get("temporal_stabilization"))
    print("anchor", manifest.get("anchor_fusion"))
    print("track_artifacts", manifest.get("track_artifacts"))
    print("lifting", manifest.get("lifting"))
    print("lift_meta", lifted.get("lift_meta"))
    print("first_valid_3d", first_valid_3d(lifted))
    print("scene_graph_4d", {
        "coord_frame": scene_graph_4d.get("coord_frame"),
        "valid_3d": scene_graph_4d.get("valid_3d"),
        "object_track_count": scene_graph_4d.get("object_track_count"),
        "frame_count": scene_graph_4d.get("frame_count"),
        "temporal_edge_count": len(scene_graph_4d.get("temporal_edges", [])),
        "spatial_relation_count": len(scene_graph_4d.get("spatial_relations", [])),
    })
    if fc2d:
        winner2d = fc2d.get("winner_by_metric") or {}
        print("forecast_2d", {
            "sample_count": fc2d.get("sample_count"),
            "best_center_model": fc2d.get("best_center_model") or winner2d.get("mean_center_l2"),
            "best_size_model": fc2d.get("best_size_model") or winner2d.get("mean_size_l1"),
            "winner_by_metric": winner2d,
        })
    if fc3d:
        winner3d = fc3d.get("winner_by_metric") or {}
        print("forecast_3d", {
            "sample_count": fc3d.get("sample_count"),
            "best_mean_world_l2_model": fc3d.get("best_mean_world_l2_model") or winner3d.get("mean_world_l2"),
            "best_max_world_l2_model": fc3d.get("best_max_world_l2_model") or winner3d.get("max_world_l2"),
            "winner_by_metric": winner3d,
        })


if __name__ == "__main__":
    main()
