from collections import defaultdict

def _avg_box(boxes):
    n = len(boxes)
    vals = [0, 0, 0, 0]
    for box in boxes:
        for i in range(4):
            vals[i] += box[i]
    return [int(v / n) for v in vals]

def smooth_observations(observations, radius=2):
    by_track = defaultdict(list)
    for obs in observations:
        by_track[obs["track_id"]].append(dict(obs))

    out = []
    for track_id, items in by_track.items():
        items.sort(key=lambda x: x["frame_idx"])
        for i, item in enumerate(items):
            lo = max(0, i - radius)
            hi = min(len(items), i + radius + 1)
            box = _avg_box([x["box"] for x in items[lo:hi]])
            new_item = dict(item)
            new_item["box"] = box
            out.append(new_item)

    out.sort(key=lambda x: (x["frame_idx"], x["track_id"]))
    return out
