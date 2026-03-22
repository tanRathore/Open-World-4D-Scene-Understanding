import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def size(box):
    x1, y1, x2, y2 = box
    return (x2 - x1, y2 - y1)


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def load_tracks(path):
    rows = json.load(open(path, "r", encoding="utf-8"))
    by_track = defaultdict(list)
    for r in rows:
        key = (str(r["track_id"]), str(r["label"]))
        by_track[key].append(r)
    for k in by_track:
        by_track[k] = sorted(by_track[k], key=lambda x: int(x["frame_idx"]))
    return by_track


def summarize_track(rows):
    if len(rows) < 2:
        return {
            "frames": len(rows),
            "mean_center_step": 0.0,
            "mean_size_step": 0.0,
            "max_center_step": 0.0,
            "max_size_step": 0.0,
        }

    center_steps = []
    size_steps = []

    prev_c = center(rows[0]["box"])
    prev_s = size(rows[0]["box"])

    for r in rows[1:]:
        cur_c = center(r["box"])
        cur_s = size(r["box"])

        center_steps.append(dist(prev_c, cur_c))
        size_steps.append(dist(prev_s, cur_s))

        prev_c = cur_c
        prev_s = cur_s

    return {
        "frames": len(rows),
        "mean_center_step": sum(center_steps) / len(center_steps),
        "mean_size_step": sum(size_steps) / len(size_steps),
        "max_center_step": max(center_steps),
        "max_size_step": max(size_steps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--stabilized", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    raw_tracks = load_tracks(args.raw)
    stab_tracks = load_tracks(args.stabilized)

    keys = sorted(set(raw_tracks) | set(stab_tracks))
    report = []

    for key in keys:
        raw_rows = raw_tracks.get(key, [])
        stab_rows = stab_tracks.get(key, [])

        raw_stats = summarize_track(raw_rows)
        stab_stats = summarize_track(stab_rows)

        row = {
            "track_id": key[0],
            "label": key[1],
            "raw_frames": raw_stats["frames"],
            "stab_frames": stab_stats["frames"],
            "raw_mean_center_step": raw_stats["mean_center_step"],
            "stab_mean_center_step": stab_stats["mean_center_step"],
            "raw_mean_size_step": raw_stats["mean_size_step"],
            "stab_mean_size_step": stab_stats["mean_size_step"],
            "raw_max_center_step": raw_stats["max_center_step"],
            "stab_max_center_step": stab_stats["max_center_step"],
            "raw_max_size_step": raw_stats["max_size_step"],
            "stab_max_size_step": stab_stats["max_size_step"],
            "center_step_delta": stab_stats["mean_center_step"] - raw_stats["mean_center_step"],
            "size_step_delta": stab_stats["mean_size_step"] - raw_stats["mean_size_step"],
        }
        report.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("wrote", out_path)
    for r in report:
        print(r)


if __name__ == "__main__":
    main()
