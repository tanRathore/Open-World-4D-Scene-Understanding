import math
from collections import defaultdict


def _has_valid_world(node):
    world = node.get("world_xyz")
    if not node.get("valid_3d", False):
        return False
    if not isinstance(world, list) or len(world) != 3:
        return False
    return all(v is not None for v in world)


def _image_relations(a, b, near_thresh):
    rels = []

    ax, ay = a["image_center_xy"]
    bx, by = b["image_center_xy"]

    if ax < bx:
        rels.append({
            "subject_track_id": a["track_id"],
            "subject_label": a["label"],
            "object_track_id": b["track_id"],
            "object_label": b["label"],
            "relation": "left_of",
            "relation_space": "image_plane",
        })

    if ay < by:
        rels.append({
            "subject_track_id": a["track_id"],
            "subject_label": a["label"],
            "object_track_id": b["track_id"],
            "object_label": b["label"],
            "relation": "above",
            "relation_space": "image_plane",
        })

    dx = ax - bx
    dy = ay - by
    if math.sqrt(dx * dx + dy * dy) <= near_thresh:
        rels.append({
            "subject_track_id": a["track_id"],
            "subject_label": a["label"],
            "object_track_id": b["track_id"],
            "object_label": b["label"],
            "relation": "near",
            "relation_space": "image_plane",
        })

    return rels


def _world_relations(a, b, near_3d_thresh=1.5, depth_order_thresh=0.25):
    rels = []

    ax, ay, az = a["world_xyz"]
    bx, by, bz = b["world_xyz"]

    dx = ax - bx
    dy = ay - by
    dz = az - bz
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    if dist <= near_3d_thresh:
        rels.append({
            "subject_track_id": a["track_id"],
            "subject_label": a["label"],
            "object_track_id": b["track_id"],
            "object_label": b["label"],
            "relation": "near_3d",
            "relation_space": "camera",
            "distance_3d": round(dist, 6),
        })

    if az + depth_order_thresh < bz:
        rels.append({
            "subject_track_id": a["track_id"],
            "subject_label": a["label"],
            "object_track_id": b["track_id"],
            "object_label": b["label"],
            "relation": "in_front_of",
            "relation_space": "camera",
            "depth_delta": round(bz - az, 6),
        })
    elif bz + depth_order_thresh < az:
        rels.append({
            "subject_track_id": b["track_id"],
            "subject_label": b["label"],
            "object_track_id": a["track_id"],
            "object_label": a["label"],
            "relation": "in_front_of",
            "relation_space": "camera",
            "depth_delta": round(az - bz, 6),
        })

    return rels


def _relation_set(nodes, near_thresh):
    rels = []
    nodes = sorted(nodes, key=lambda x: str(x["track_id"]))

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a = nodes[i]
            b = nodes[j]

            rels.extend(_image_relations(a, b, near_thresh))

            if _has_valid_world(a) and _has_valid_world(b):
                rels.extend(_world_relations(a, b))

    return rels


def build_scene_graph_4d(object_state_4d, near_thresh=140):
    frames = defaultdict(list)
    temporal_edges = []

    for obj in object_state_4d.get("objects", []):
        states = obj.get("states", [])

        for s in states:
            frames[int(s["frame_idx"])].append({
                "track_id": obj["track_id"],
                "label": obj["label"],
                "image_center_xy": s["image_center_xy"],
                "image_center_xy_norm": s["image_center_xy_norm"],
                "image_size_wh": s["image_size_wh"],
                "image_size_wh_norm": s["image_size_wh_norm"],
                "velocity_xy": s["velocity_xy"],
                "speed_2d": s["speed_2d"],
                "depth": s["depth"],
                "world_xyz": s["world_xyz"],
                "world_velocity_xyz": s["world_velocity_xyz"],
                "bbox_3d": s["bbox_3d"],
                "valid_3d": s["valid_3d"],
                "coord_frame": s["coord_frame"],
                "gap_filled": s["gap_filled"],
                "score": s["score"],
            })

        for i in range(1, len(states)):
            prev_s = states[i - 1]
            cur_s = states[i]
            temporal_edges.append({
                "track_id": obj["track_id"],
                "label": obj["label"],
                "from_frame": int(prev_s["frame_idx"]),
                "to_frame": int(cur_s["frame_idx"]),
                "dt": int(cur_s["frame_idx"]) - int(prev_s["frame_idx"]),
                "velocity_xy": cur_s["velocity_xy"],
                "speed_2d": cur_s["speed_2d"],
                "world_velocity_xyz": cur_s["world_velocity_xyz"],
                "valid_3d": cur_s["valid_3d"],
            })

    frame_graphs = []
    for frame_idx in sorted(frames):
        nodes = sorted(frames[frame_idx], key=lambda x: str(x["track_id"]))
        spatial_edges = _relation_set(nodes, near_thresh=near_thresh)
        frame_graphs.append({
            "frame_idx": int(frame_idx),
            "nodes": nodes,
            "spatial_edges": spatial_edges,
        })

    return {
        "representation": "scene_graph_4d",
        "coord_frame": object_state_4d.get("coord_frame", "image_plane"),
        "valid_3d": bool(object_state_4d.get("valid_3d", False)),
        "object_track_count": int(object_state_4d.get("track_count", 0)),
        "frame_count": len(frame_graphs),
        "frames": frame_graphs,
        "temporal_edges": temporal_edges,
    }
