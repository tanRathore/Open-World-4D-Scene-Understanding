from pathlib import Path
import sys
import json
import csv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ow4d.io import load_json, save_json
from ow4d.pipeline import run_core
from ow4d.metrics import compare_track_rows, summarize_compare_reports


def summarize_scene_graph_4d(scene_graph_4d):
    frames = scene_graph_4d.get("frames", [])
    if not frames:
        return {
            "object_track_count": int(scene_graph_4d.get("object_track_count", 0)),
            "frame_count": 0,
            "temporal_edge_count": len(scene_graph_4d.get("temporal_edges", [])),
            "mean_nodes_per_frame": 0.0,
            "max_nodes_per_frame": 0,
            "mean_spatial_edges_per_frame": 0.0,
            "max_spatial_edges_per_frame": 0,
            "is_multi_object": False,
        }

    node_counts = [len(f.get("nodes", [])) for f in frames]
    edge_counts = [len(f.get("spatial_edges", [])) for f in frames]

    return {
        "object_track_count": int(scene_graph_4d.get("object_track_count", 0)),
        "frame_count": len(frames),
        "temporal_edge_count": len(scene_graph_4d.get("temporal_edges", [])),
        "mean_nodes_per_frame": sum(node_counts) / len(node_counts),
        "max_nodes_per_frame": max(node_counts),
        "mean_spatial_edges_per_frame": sum(edge_counts) / len(edge_counts),
        "max_spatial_edges_per_frame": max(edge_counts),
        "is_multi_object": max(node_counts) > 1,
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    spec = json.load(open(args.spec, "r", encoding="utf-8"))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    per_sequence = {}

    for item in spec["sequences"]:
        name = item["name"]
        seq_out = out_root / name

        print("run", name)
        result = run_core(
            input_path=item["input"],
            prompt_text=item["text"],
            out_dir=str(seq_out),
            adapter="grounded_sam2",
            pred_path=item["pred_path"],
            anchor_path=item.get("anchor_path"),
            stride=args.stride,
        )

        raw_rows = load_json(seq_out / "tracks_raw.json")
        stab_rows = load_json(seq_out / "tracks_stabilized.json")
        final_rows = load_json(seq_out / "tracks.json")

        raw_to_stab_reports = compare_track_rows(raw_rows, stab_rows)
        raw_to_final_reports = compare_track_rows(raw_rows, final_rows)

        raw_to_stab_summary = summarize_compare_reports(raw_to_stab_reports)
        raw_to_final_summary = summarize_compare_reports(raw_to_final_reports)

        manifest = load_json(seq_out / "manifest.json")
        forecast_compare = load_json(seq_out / "forecast_baseline_compare.json")
        future_graph_compare = load_json(seq_out / "future_graph_baseline_compare.json")
        scene_dynamics = load_json(seq_out / "scene_dynamics.json")
        scene_graph_4d = load_json(seq_out / "scene_graph_4d.json")

        sg4d_summary = summarize_scene_graph_4d(scene_graph_4d)

        seq_summary = {
            "name": name,
            "prompt_text": item["text"],
            "track_count": int(result["track_count"]),
            "obs_count": int(result["obs_count"]),
            "raw_to_stabilized": raw_to_stab_summary,
            "raw_to_final": raw_to_final_summary,
            "anchor_fusion": manifest.get("anchor_fusion", {}),
            "scene_graph_4d": sg4d_summary,
            "relation_transition_count": int(scene_dynamics.get("relation_transition_count", 0)),
            "forecast_compare": forecast_compare,
            "future_graph_compare": future_graph_compare,
        }
        per_sequence[name] = seq_summary

        rows.append({
            "sequence": name,
            "track_count": int(result["track_count"]),
            "obs_count": int(result["obs_count"]),
            "is_multi_object": sg4d_summary["is_multi_object"],
            "mean_nodes_per_frame": round(sg4d_summary["mean_nodes_per_frame"], 6),
            "max_nodes_per_frame": int(sg4d_summary["max_nodes_per_frame"]),
            "mean_spatial_edges_per_frame": round(sg4d_summary["mean_spatial_edges_per_frame"], 6),
            "max_spatial_edges_per_frame": int(sg4d_summary["max_spatial_edges_per_frame"]),
            "relation_transition_count": int(scene_dynamics.get("relation_transition_count", 0)),
            "stab_center_delta": round(raw_to_stab_summary["mean_center_step_delta"], 6),
            "stab_size_delta": round(raw_to_stab_summary["mean_size_step_delta"], 6),
            "final_center_delta": round(raw_to_final_summary["mean_center_step_delta"], 6),
            "final_size_delta": round(raw_to_final_summary["mean_size_step_delta"], 6),
            "anchor_fused_rows": int(manifest.get("anchor_fusion", {}).get("fused_rows", 0)),
            "anchor_matched_rows": int(manifest.get("anchor_fusion", {}).get("matched_rows", 0)),
            "anchor_rejected_rows": int(manifest.get("anchor_fusion", {}).get("rejected_rows", 0)),
            "forecast_best_center": forecast_compare["winner_by_metric"]["mean_center_l2"],
            "forecast_best_size": forecast_compare["winner_by_metric"]["mean_size_l1"],
            "future_graph_best_center": future_graph_compare["winner_by_metric"]["mean_node_center_l2"],
            "future_graph_best_size": future_graph_compare["winner_by_metric"]["mean_node_size_l1"],
            "future_graph_best_relation": future_graph_compare["winner_by_metric"]["mean_relation_jaccard"],
        })

    summary = {
        "spec": args.spec,
        "out_root": str(out_root),
        "sequence_count": len(rows),
        "rows": rows,
        "per_sequence": per_sequence,
    }
    save_json(out_root / "summary.json", summary)

    csv_path = out_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("wrote", out_root / "summary.json")
    print("wrote", csv_path)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
