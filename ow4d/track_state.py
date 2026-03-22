import math
from collections import defaultdict


def _center(box):
    x1, y1, x2, y2 = box
    return [round((x1 + x2) / 2.0, 3), round((y1 + y2) / 2.0, 3)]


def _size(box):
    x1, y1, x2, y2 = box
    return [int(x2 - x1), int(y2 - y1)]


def build_track_state(rows):
    by_track = defaultdict(list)
    for row in rows:
        key = (str(row["track_id"]), str(row.get("label", "")))
        by_track[key].append(row)

    tracks = []

    for (track_id, label), group in sorted(by_track.items()):
        group = sorted(group, key=lambda x: int(x["frame_idx"]))
        states = []
        prev_center = None
        prev_size = None
        speeds = []

        for row in group:
            box = row["box"]
            center = _center(box)
            size = _size(box)

            if prev_center is None:
                velocity = [0.0, 0.0]
                speed = 0.0
            else:
                velocity = [
                    round(center[0] - prev_center[0], 3),
                    round(center[1] - prev_center[1], 3),
                ]
                speed = round(math.sqrt(velocity[0] ** 2 + velocity[1] ** 2), 3)

            if prev_size is None:
                size_velocity = [0, 0]
            else:
                size_velocity = [
                    int(size[0] - prev_size[0]),
                    int(size[1] - prev_size[1]),
                ]

            state = {
                "frame_idx": int(row["frame_idx"]),
                "box": box,
                "center": center,
                "size": size,
                "velocity": velocity,
                "speed": speed,
                "size_velocity": size_velocity,
                "score": float(row.get("score", 0.0)),
                "gap_filled": bool(row.get("gap_filled", False)),
            }
            states.append(state)
            speeds.append(speed)
            prev_center = center
            prev_size = size

        track = {
            "track_id": track_id,
            "label": label,
            "start_frame": int(group[0]["frame_idx"]),
            "end_frame": int(group[-1]["frame_idx"]),
            "num_frames": len(group),
            "mean_speed": round(sum(speeds) / len(speeds), 3) if speeds else 0.0,
            "max_speed": round(max(speeds), 3) if speeds else 0.0,
            "states": states,
        }
        tracks.append(track)

    return {
        "track_count": len(tracks),
        "tracks": tracks,
    }
