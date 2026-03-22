def build_dynamic_object_graph(track_state, scene_dynamics):
    tracks = track_state.get("tracks", [])
    dynamics = scene_dynamics or {}

    object_nodes = []
    temporal_edges = []

    for track in tracks:
        node = {
            "track_id": track["track_id"],
            "label": track["label"],
            "start_frame": track["start_frame"],
            "end_frame": track["end_frame"],
            "num_frames": track["num_frames"],
            "mean_speed": track["mean_speed"],
            "max_speed": track["max_speed"],
        }
        object_nodes.append(node)

        states = track.get("states", [])
        for i in range(1, len(states)):
            prev = states[i - 1]
            cur = states[i]
            temporal_edges.append({
                "track_id": track["track_id"],
                "label": track["label"],
                "from_frame": int(prev["frame_idx"]),
                "to_frame": int(cur["frame_idx"]),
                "dt": int(cur["frame_idx"]) - int(prev["frame_idx"]),
                "from_center": prev["center"],
                "to_center": cur["center"],
                "velocity": cur["velocity"],
                "speed": cur["speed"],
                "size_velocity": cur["size_velocity"],
                "gap_filled_from": bool(prev.get("gap_filled", False)),
                "gap_filled_to": bool(cur.get("gap_filled", False)),
            })

    relation_events = []
    for item in dynamics.get("relation_transitions", []):
        relation_events.append({
            "frame_idx": int(item["frame_idx"]),
            "subject_track_id": item["subject_track_id"],
            "object_track_id": item["object_track_id"],
            "from_relations": item["from_relations"],
            "to_relations": item["to_relations"],
            "event_type": "relation_change",
        })

    return {
        "object_node_count": len(object_nodes),
        "temporal_edge_count": len(temporal_edges),
        "relation_event_count": len(relation_events),
        "object_nodes": object_nodes,
        "temporal_edges": temporal_edges,
        "relation_events": relation_events,
    }
