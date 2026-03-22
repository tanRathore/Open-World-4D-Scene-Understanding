import math
from collections import defaultdict


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _size(box):
    x1, y1, x2, y2 = box
    return (x2 - x1, y2 - y1)


def _track_key(row):
    return (str(row.get("track_id", "")), str(row.get("label", "")))


def _group_rows(rows):
    out = defaultdict(list)
    for row in rows:
        out[_track_key(row)].append(row)
    for key in out:
        out[key] = sorted(out[key], key=lambda x: int(x["frame_idx"]))
    return out


def _step_stats(rows):
    rows = sorted(rows, key=lambda x: int(x["frame_idx"]))
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

    for i in range(1, len(rows)):
        prev = rows[i - 1]
        cur = rows[i]

        pcx, pcy = _center(prev["box"])
        ccx, ccy = _center(cur["box"])
        center_steps.append(math.sqrt((ccx - pcx) ** 2 + (ccy - pcy) ** 2))

        pw, ph = _size(prev["box"])
        cw, ch = _size(cur["box"])
        size_steps.append(math.sqrt((cw - pw) ** 2 + (ch - ph) ** 2))

    return {
        "frames": len(rows),
        "mean_center_step": sum(center_steps) / len(center_steps),
        "mean_size_step": sum(size_steps) / len(size_steps),
        "max_center_step": max(center_steps),
        "max_size_step": max(size_steps),
    }


def compare_track_rows(raw_rows, other_rows):
    raw_by_track = _group_rows(raw_rows)
    other_by_track = _group_rows(other_rows)

    reports = []
    for key in sorted(set(raw_by_track) & set(other_by_track)):
        raw_stats = _step_stats(raw_by_track[key])
        other_stats = _step_stats(other_by_track[key])

        track_id, label = key
        reports.append({
            "track_id": track_id,
            "label": label,
            "raw_frames": raw_stats["frames"],
            "other_frames": other_stats["frames"],
            "raw_mean_center_step": raw_stats["mean_center_step"],
            "other_mean_center_step": other_stats["mean_center_step"],
            "raw_mean_size_step": raw_stats["mean_size_step"],
            "other_mean_size_step": other_stats["mean_size_step"],
            "raw_max_center_step": raw_stats["max_center_step"],
            "other_max_center_step": other_stats["max_center_step"],
            "raw_max_size_step": raw_stats["max_size_step"],
            "other_max_size_step": other_stats["max_size_step"],
            "center_step_delta": other_stats["mean_center_step"] - raw_stats["mean_center_step"],
            "size_step_delta": other_stats["mean_size_step"] - raw_stats["mean_size_step"],
        })

    return reports


def summarize_compare_reports(reports):
    if not reports:
        return {
            "track_count": 0,
            "mean_center_step_delta": 0.0,
            "mean_size_step_delta": 0.0,
            "mean_raw_center_step": 0.0,
            "mean_other_center_step": 0.0,
            "mean_raw_size_step": 0.0,
            "mean_other_size_step": 0.0,
        }

    n = len(reports)
    return {
        "track_count": n,
        "mean_center_step_delta": sum(r["center_step_delta"] for r in reports) / n,
        "mean_size_step_delta": sum(r["size_step_delta"] for r in reports) / n,
        "mean_raw_center_step": sum(r["raw_mean_center_step"] for r in reports) / n,
        "mean_other_center_step": sum(r["other_mean_center_step"] for r in reports) / n,
        "mean_raw_size_step": sum(r["raw_mean_size_step"] for r in reports) / n,
        "mean_other_size_step": sum(r["other_mean_size_step"] for r in reports) / n,
    }
