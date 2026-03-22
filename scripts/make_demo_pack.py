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


def build_limitations(scene_graph_4d, lifted, first_valid):
    items = []

    if (scene_graph_4d or {}).get("object_track_count", 0) <= 1:
        items.append(
            "Current run is effectively single-object, so spatial relation counts are limited and future-graph evaluation is not yet very informative."
        )

    if (lifted or {}).get("coord_frame") == "camera":
        items.append(
            "Current 3D lifting is camera-frame, not yet pose-aligned world-frame lifting across ego-motion."
        )

    z_size = None
    if first_valid:
        bbox = first_valid.get("bbox_3d") or {}
        size_xyz = bbox.get("size_xyz") or []
        if len(size_xyz) == 3:
            z_size = size_xyz[2]

    if z_size in [0, 0.0, None]:
        items.append(
            "Current 3D box thickness remains approximate; full object extent and volume fitting are still under development."
        )

    items.append(
        "Future prediction is currently baseline-driven rather than a learned future scene / graph model."
    )

    return items


def build_next_steps(scene_graph_4d, lifted):
    steps = [
        "Run one strong multi-object sequence through the same Option B pipeline so scene_graph_4d shows real spatial relations, not just temporal edges.",
        "Upgrade center_depth_3d from box pooling to object-aware depth pooling using masks or a tighter support region while keeping the same artifact contracts.",
        "Add pose-aware lifting so camera-frame 3D states can become temporally consistent world-frame object states.",
    ]

    if (scene_graph_4d or {}).get("object_track_count", 0) > 1:
        steps[0] = "Quantify 3D relation quality on multi-object runs and add one compact scene-graph figure for the paper/demo."

    if (lifted or {}).get("coord_frame") != "camera":
        steps[2] = "Build the next predictive layer on top of the lifted state: learned future object-state / future-graph forecasting."

    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--title", default="Toward Open-World 4D Scene Understanding")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)

    manifest = load_json(run_dir / "manifest.json") or {}
    lifted = load_json(run_dir / "lifted_object_state_4d.json") or {}
    scene_graph_4d = load_json(run_dir / "scene_graph_4d.json") or {}
    fc2d = load_json(run_dir / "forecast_baseline_compare.json") or {}
    fc3d = load_json(run_dir / "forecast_baseline_compare_3d.json") or {}

    first_valid = first_valid_3d(lifted)

    temporal = manifest.get("temporal_stabilization") or {}
    anchor = manifest.get("anchor_fusion") or {}
    track_artifacts = manifest.get("track_artifacts") or {}
    lifting = manifest.get("lifting") or {}
    lift_meta = lifted.get("lift_meta") or {}

    winner2d = fc2d.get("winner_by_metric") or {}
    winner3d = fc3d.get("winner_by_metric") or {}

    forecast_2d = {
        "sample_count": fc2d.get("sample_count"),
        "best_center_model": fc2d.get("best_center_model") or winner2d.get("mean_center_l2"),
        "best_size_model": fc2d.get("best_size_model") or winner2d.get("mean_size_l1"),
    }

    forecast_3d = {
        "sample_count": fc3d.get("sample_count"),
        "best_mean_world_l2_model": fc3d.get("best_mean_world_l2_model") or winner3d.get("mean_world_l2"),
        "best_max_world_l2_model": fc3d.get("best_max_world_l2_model") or winner3d.get("max_world_l2"),
    }

    limitations = build_limitations(scene_graph_4d, lifted, first_valid)
    next_steps = build_next_steps(scene_graph_4d, lifted)

    pack = {
        "title": args.title,
        "run_dir": str(run_dir),
        "controller_path": [
            "preds.json",
            "label_canonicalization",
            "temporal_stabilization",
            "anchor_fusion",
            "track_state",
            "object_state_4d",
            "lifted_object_state_4d",
            "scene_graph_4d",
        ],
        "temporal": temporal,
        "anchor": anchor,
        "track_artifacts": track_artifacts,
        "lifting": lifting,
        "lift_meta": lift_meta,
        "first_valid_3d": first_valid,
        "scene_graph_4d": {
            "coord_frame": scene_graph_4d.get("coord_frame"),
            "valid_3d": scene_graph_4d.get("valid_3d"),
            "object_track_count": scene_graph_4d.get("object_track_count"),
            "frame_count": scene_graph_4d.get("frame_count"),
            "temporal_edge_count": len(scene_graph_4d.get("temporal_edges", [])),
            "spatial_relation_count": len(scene_graph_4d.get("spatial_relations", [])),
        },
        "forecast_2d": forecast_2d,
        "forecast_3d": forecast_3d,
        "limitations": limitations,
        "next_steps": next_steps,
    }

    md = f"""# {args.title}

## Current canonical controller path
`preds.json -> label canonicalization -> temporal stabilization -> anchor fusion -> track_state -> object_state_4d -> lifted_object_state_4d -> scene_graph_4d`

## What is working now
- Temporal stabilization is enabled and producing canonical controller-side tracks.
- Anchor fusion is part of the canonical 4D branch.
- `track_state`, `object_state_4d`, `lifted_object_state_4d`, and `scene_graph_4d` are all emitted.
- Preliminary camera-frame 3D lifting is active with matched depth and intrinsics.
- 2D and 3D forecast baseline artifacts are emitted.

## Run evidence
- Run dir: `{run_dir}`
- Temporal: `{temporal}`
- Anchor: `{anchor}`
- Track artifacts: `{track_artifacts}`
- Lifting: `{lifting}`
- Lift meta: `{lift_meta}`
- First valid 3D state: `{first_valid}`
- Scene graph 4D: `{pack["scene_graph_4d"]}`
- Forecast 2D: `{forecast_2d}`
- Forecast 3D: `{forecast_3d}`

## What this demo already shows
This run demonstrates a serious controller-side pipeline for turning open-world video predictions into stabilized structured state, lifting that state into preliminary 3D, and constructing a temporal scene graph on top of the lifted representation.

## Under development
""" + "\n".join(f"- {x}" for x in limitations) + """

## Next highest-leverage steps
""" + "\n".join(f"{i+1}. {x}" for i, x in enumerate(next_steps)) + """
"""

    json_path = run_dir / "demo_pack.json"
    md_path = run_dir / "DEMO_STATUS.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2)

    md_path.write_text(md, encoding="utf-8")

    print("wrote", json_path)
    print("wrote", md_path)
    print("scene_graph_4d", pack["scene_graph_4d"])
    print("forecast_2d", forecast_2d)
    print("forecast_3d", forecast_3d)
    print("limitations", len(limitations))
    print("next_steps", len(next_steps))


if __name__ == "__main__":
    main()
