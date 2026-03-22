from collections import defaultdict

def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def _dist(a, b):
    ax, ay = a
    bx, by = b
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

def build_scene_graph(observations, near_thresh=140):
    by_frame = defaultdict(list)
    for obs in observations:
        by_frame[obs["frame_idx"]].append(obs)

    frames = []
    for frame_idx in sorted(by_frame):
        objs = by_frame[frame_idx]
        nodes = []
        edges = []

        for obj in objs:
            cx, cy = _center(obj["box"])
            nodes.append({
                "track_id": obj["track_id"],
                "label": obj["label"],
                "score": obj["score"],
                "box": obj["box"],
                "center": [cx, cy]
            })

        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                a = objs[i]
                b = objs[j]
                ca = _center(a["box"])
                cb = _center(b["box"])
                d = _dist(ca, cb)

                if d <= near_thresh:
                    edges.append({
                        "src": a["track_id"],
                        "dst": b["track_id"],
                        "relation": "near",
                        "value": round(d, 2)
                    })

                if ca[0] < cb[0]:
                    edges.append({
                        "src": a["track_id"],
                        "dst": b["track_id"],
                        "relation": "left_of",
                        "value": round(cb[0] - ca[0], 2)
                    })
                else:
                    edges.append({
                        "src": b["track_id"],
                        "dst": a["track_id"],
                        "relation": "left_of",
                        "value": round(ca[0] - cb[0], 2)
                    })

                if ca[1] < cb[1]:
                    edges.append({
                        "src": a["track_id"],
                        "dst": b["track_id"],
                        "relation": "above",
                        "value": round(cb[1] - ca[1], 2)
                    })
                else:
                    edges.append({
                        "src": b["track_id"],
                        "dst": a["track_id"],
                        "relation": "above",
                        "value": round(ca[1] - cb[1], 2)
                    })

        frames.append({
            "frame_idx": frame_idx,
            "nodes": nodes,
            "edges": edges
        })

    return {"frames": frames}
