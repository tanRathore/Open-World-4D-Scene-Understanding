from collections import defaultdict

def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def _area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def summarize_tracks(observations):
    by_track = defaultdict(list)
    for obs in observations:
        by_track[obs["track_id"]].append(obs)

    rows = []
    for track_id, items in sorted(by_track.items()):
        items = sorted(items, key=lambda x: x["frame_idx"])

        motion = 0.0
        centers = [_center(x["box"]) for x in items]
        for i in range(1, len(centers)):
            ax, ay = centers[i - 1]
            bx, by = centers[i]
            motion += ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

        rows.append({
            "track_id": track_id,
            "label": items[0]["label"],
            "source_prompt": items[0]["source_prompt"],
            "frames_seen": len(items),
            "start_frame": items[0]["frame_idx"],
            "end_frame": items[-1]["frame_idx"],
            "avg_score": round(sum(x["score"] for x in items) / len(items), 4),
            "avg_area": round(sum(_area(x["box"]) for x in items) / len(items), 2),
            "total_motion_px": round(motion, 2),
        })

    return rows
