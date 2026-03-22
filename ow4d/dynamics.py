import math
from collections import defaultdict


def _relation_set(a_center, b_center, near_thresh):
    rels = set()
    if a_center[0] < b_center[0]:
        rels.add("left_of")
    if a_center[1] < b_center[1]:
        rels.add("above")
    dx = a_center[0] - b_center[0]
    dy = a_center[1] - b_center[1]
    if math.sqrt(dx * dx + dy * dy) <= near_thresh:
        rels.add("near")
    return rels


def build_scene_dynamics(track_state, near_thresh=140):
    tracks = track_state.get("tracks", [])

    summaries = []
    states_by_frame = defaultdict(list)

    for track in tracks:
        states = track.get("states", [])
        if not states:
            continue

        first = states[0]
        last = states[-1]
        dx = last["center"][0] - first["center"][0]
        dy = last["center"][1] - first["center"][1]
        net_disp = round(math.sqrt(dx * dx + dy * dy), 3)

        moving_frames = sum(1 for s in states if float(s.get("speed", 0.0)) > 0.5)
        moving_ratio = round(moving_frames / max(len(states), 1), 3)

        if net_disp < 5:
            motion_label = "static"
        elif abs(dx) >= abs(dy):
            motion_label = "horizontal"
        else:
            motion_label = "vertical"

        summaries.append({
            "track_id": track["track_id"],
            "label": track["label"],
            "num_frames": track["num_frames"],
            "start_frame": track["start_frame"],
            "end_frame": track["end_frame"],
            "mean_speed": track["mean_speed"],
            "max_speed": track["max_speed"],
            "net_displacement": net_disp,
            "delta_center": [round(dx, 3), round(dy, 3)],
            "moving_ratio": moving_ratio,
            "motion_label": motion_label,
        })

        for state in states:
            states_by_frame[int(state["frame_idx"])].append({
                "track_id": track["track_id"],
                "label": track["label"],
                "center": state["center"],
                "size": state["size"],
            })

    relation_frames = []
    relation_transitions = []
    prev_pairs = {}

    for frame_idx in sorted(states_by_frame):
        items = sorted(states_by_frame[frame_idx], key=lambda x: str(x["track_id"]))
        frame_rels = []
        cur_pairs = {}

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a = items[i]
                b = items[j]
                pair_key = (str(a["track_id"]), str(b["track_id"]))
                rels = sorted(_relation_set(a["center"], b["center"], near_thresh))
                cur_pairs[pair_key] = rels
                frame_rels.append({
                    "subject_track_id": a["track_id"],
                    "subject_label": a["label"],
                    "object_track_id": b["track_id"],
                    "object_label": b["label"],
                    "relations": rels,
                })

                old = prev_pairs.get(pair_key)
                if old is not None and old != rels:
                    relation_transitions.append({
                        "frame_idx": frame_idx,
                        "subject_track_id": a["track_id"],
                        "object_track_id": b["track_id"],
                        "from_relations": old,
                        "to_relations": rels,
                    })

        relation_frames.append({
            "frame_idx": frame_idx,
            "relations": frame_rels,
        })
        prev_pairs = cur_pairs

    return {
        "track_count": len(summaries),
        "track_summaries": summaries,
        "frame_relation_count": len(relation_frames),
        "relation_transition_count": len(relation_transitions),
        "relation_frames": relation_frames,
        "relation_transitions": relation_transitions,
    }
