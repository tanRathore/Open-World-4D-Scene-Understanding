import math


def _node_center(n):
    if "center" in n:
        return list(n["center"])
    return list(n["image_center_xy"])


def _node_size(n):
    if "size" in n:
        return list(n["size"])
    return list(n["image_size_wh"])


def _node_velocity(n):
    if "velocity" in n:
        return list(n["velocity"])
    return list(n.get("velocity_xy", [0.0, 0.0]))


def _node_speed(n):
    if "speed" in n:
        return float(n["speed"])
    return float(n.get("speed_2d", 0.0))


def _relation_set(nodes, near_thresh):
    rels = set()
    nodes = sorted(nodes, key=lambda x: str(x["track_id"]))

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a = nodes[i]
            b = nodes[j]
            ax, ay = _node_center(a)
            bx, by = _node_center(b)

            if ax < bx:
                rels.add((str(a["track_id"]), str(b["track_id"]), "left_of", "image_plane"))
            if ay < by:
                rels.add((str(a["track_id"]), str(b["track_id"]), "above", "image_plane"))

            dx = ax - bx
            dy = ay - by
            if math.sqrt(dx * dx + dy * dy) <= near_thresh:
                rels.add((str(a["track_id"]), str(b["track_id"]), "near", "image_plane"))

    return rels


def _center_err(a, b):
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.sqrt(dx * dx + dy * dy)


def _size_err(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _eval_graph_predictions(predictions):
    center_errs = []
    size_errs = []
    rel_jaccards = []
    rel_exact = []

    for item in predictions:
        for pred_graph, tgt_graph in zip(item["prediction_graphs"], item["target_graphs"]):
            pred_nodes = {str(n["track_id"]): n for n in pred_graph["nodes"]}
            tgt_nodes = {str(n["track_id"]): n for n in tgt_graph["nodes"]}

            shared = sorted(set(pred_nodes) & set(tgt_nodes))
            for tid in shared:
                pn = pred_nodes[tid]
                tn = tgt_nodes[tid]
                center_errs.append(_center_err(_node_center(pn), _node_center(tn)))
                size_errs.append(_size_err(_node_size(pn), _node_size(tn)))

            pred_rel = set()
            for rel in pred_graph["relations"]:
                pred_rel.add((
                    str(rel["subject_track_id"]),
                    str(rel["object_track_id"]),
                    str(rel["relation"]),
                    str(rel.get("relation_space", "unknown")),
                ))

            tgt_rel = set()
            for rel in tgt_graph["relations"]:
                tgt_rel.add((
                    str(rel["subject_track_id"]),
                    str(rel["object_track_id"]),
                    str(rel["relation"]),
                    str(rel.get("relation_space", "unknown")),
                ))

            union = pred_rel | tgt_rel
            inter = pred_rel & tgt_rel

            if not union:
                rel_jaccards.append(1.0)
                rel_exact.append(1.0)
            else:
                rel_jaccards.append(len(inter) / len(union))
                rel_exact.append(1.0 if pred_rel == tgt_rel else 0.0)

    return {
        "sample_count": len(predictions),
        "mean_node_center_l2": round(sum(center_errs) / max(len(center_errs), 1), 4),
        "max_node_center_l2": round(max(center_errs), 4) if center_errs else 0.0,
        "mean_node_size_l1": round(sum(size_errs) / max(len(size_errs), 1), 4),
        "max_node_size_l1": round(max(size_errs), 4) if size_errs else 0.0,
        "mean_relation_jaccard": round(sum(rel_jaccards) / max(len(rel_jaccards), 1), 4),
        "relation_frame_exact_rate": round(sum(rel_exact) / max(len(rel_exact), 1), 4),
    }


def _make_graph(nodes, frame_idx, near_thresh):
    rels = []
    rel_set = _relation_set(nodes, near_thresh)

    for subj, obj, rel, rel_space in sorted(rel_set):
        rels.append({
            "subject_track_id": subj,
            "object_track_id": obj,
            "relation": rel,
            "relation_space": rel_space,
        })

    return {
        "frame_idx": int(frame_idx),
        "nodes": nodes,
        "relations": rels,
    }


def build_constant_position_future_graph_baseline(future_graph_samples, near_thresh=140):
    samples = future_graph_samples.get("samples", [])
    predictions = []

    for sample in samples:
        obs_graphs = sample["obs_graphs"]
        tgt_graphs = sample["target_graphs"]
        if not obs_graphs or not tgt_graphs:
            continue

        last_obs = obs_graphs[-1]
        seed_nodes = []
        for n in last_obs["nodes"]:
            seed_nodes.append({
                "track_id": n["track_id"],
                "label": n["label"],
                "center": _node_center(n),
                "size": _node_size(n),
                "velocity": [0.0, 0.0],
                "speed": 0.0,
                "gap_filled": bool(n.get("gap_filled", False)),
            })

        pred_graphs = []
        for tgt in tgt_graphs:
            nodes = []
            for n in seed_nodes:
                nodes.append({
                    "track_id": n["track_id"],
                    "label": n["label"],
                    "center": list(n["center"]),
                    "size": list(n["size"]),
                    "velocity": [0.0, 0.0],
                    "speed": 0.0,
                    "gap_filled": bool(n.get("gap_filled", False)),
                })
            pred_graphs.append(_make_graph(nodes, tgt["frame_idx"], near_thresh))

        predictions.append({
            "obs_start_frame": sample["obs_start_frame"],
            "obs_end_frame": sample["obs_end_frame"],
            "pred_end_frame": sample["pred_end_frame"],
            "prediction_graphs": pred_graphs,
            "target_graphs": tgt_graphs,
        })

    return {
        "baseline": "constant_position_graph",
        "metrics": _eval_graph_predictions(predictions),
        "predictions": predictions,
    }


def build_hybrid_future_graph_baseline(future_graph_samples, near_thresh=140):
    samples = future_graph_samples.get("samples", [])
    predictions = []

    for sample in samples:
        obs_graphs = sample["obs_graphs"]
        tgt_graphs = sample["target_graphs"]
        if not obs_graphs or not tgt_graphs:
            continue

        last_obs = obs_graphs[-1]
        seed_nodes = []
        for n in last_obs["nodes"]:
            seed_nodes.append({
                "track_id": n["track_id"],
                "label": n["label"],
                "center": _node_center(n),
                "size": _node_size(n),
                "velocity": _node_velocity(n),
                "speed": _node_speed(n),
                "gap_filled": bool(n.get("gap_filled", False)),
            })

        pred_graphs = []
        cur_nodes = []
        for n in seed_nodes:
            cur_nodes.append({
                "track_id": n["track_id"],
                "label": n["label"],
                "center": list(n["center"]),
                "size": list(n["size"]),
                "velocity": list(n["velocity"]),
                "speed": float(n["speed"]),
                "gap_filled": bool(n.get("gap_filled", False)),
            })

        for tgt in tgt_graphs:
            next_nodes = []
            for n in cur_nodes:
                vx, vy = n["velocity"]
                cx, cy = n["center"]
                next_nodes.append({
                    "track_id": n["track_id"],
                    "label": n["label"],
                    "center": [round(cx + vx, 3), round(cy + vy, 3)],
                    "size": list(n["size"]),
                    "velocity": list(n["velocity"]),
                    "speed": float(n["speed"]),
                    "gap_filled": bool(n.get("gap_filled", False)),
                })
            pred_graphs.append(_make_graph(next_nodes, tgt["frame_idx"], near_thresh))
            cur_nodes = next_nodes

        predictions.append({
            "obs_start_frame": sample["obs_start_frame"],
            "obs_end_frame": sample["obs_end_frame"],
            "pred_end_frame": sample["pred_end_frame"],
            "prediction_graphs": pred_graphs,
            "target_graphs": tgt_graphs,
        })

    return {
        "baseline": "hybrid_future_graph",
        "metrics": _eval_graph_predictions(predictions),
        "predictions": predictions,
    }


def build_future_graph_baseline_compare(cp_graph, hybrid_graph):
    cp = cp_graph["metrics"]
    hy = hybrid_graph["metrics"]

    higher_better = {"mean_relation_jaccard", "relation_frame_exact_rate"}

    winner_by_metric = {}
    for metric in [
        "mean_node_center_l2",
        "max_node_center_l2",
        "mean_node_size_l1",
        "max_node_size_l1",
        "mean_relation_jaccard",
        "relation_frame_exact_rate",
    ]:
        if metric in higher_better:
            winner_by_metric[metric] = (
                "hybrid_future_graph" if hy[metric] >= cp[metric] else "constant_position_graph"
            )
        else:
            winner_by_metric[metric] = (
                "hybrid_future_graph" if hy[metric] <= cp[metric] else "constant_position_graph"
            )

    return {
        "sample_count": hy["sample_count"],
        "baselines": {
            "constant_position_graph": cp,
            "hybrid_future_graph": hy,
        },
        "winner_by_metric": winner_by_metric,
        "delta_hybrid_minus_cp": {
            "mean_node_center_l2": round(hy["mean_node_center_l2"] - cp["mean_node_center_l2"], 4),
            "max_node_center_l2": round(hy["max_node_center_l2"] - cp["max_node_center_l2"], 4),
            "mean_node_size_l1": round(hy["mean_node_size_l1"] - cp["mean_node_size_l1"], 4),
            "max_node_size_l1": round(hy["max_node_size_l1"] - cp["max_node_size_l1"], 4),
            "mean_relation_jaccard": round(hy["mean_relation_jaccard"] - cp["mean_relation_jaccard"], 4),
            "relation_frame_exact_rate": round(hy["relation_frame_exact_rate"] - cp["relation_frame_exact_rate"], 4),
        },
    }
