import math
from collections import defaultdict


def lerp(a, b, t):
    return a + (b - a) * t


def interp_box(box_a, box_b, t):
    return [int(round(lerp(box_a[i], box_b[i], t))) for i in range(4)]


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def clamp_box_step(prev_box, cur_box, max_center_step):
    pc = center(prev_box)
    cc = center(cur_box)

    dx = cc[0] - pc[0]
    dy = cc[1] - pc[1]
    d = math.sqrt(dx * dx + dy * dy)

    if d <= max_center_step or d == 0:
        return cur_box

    scale = max_center_step / d
    ndx = dx * scale
    ndy = dy * scale

    w = cur_box[2] - cur_box[0]
    h = cur_box[3] - cur_box[1]
    ncx = pc[0] + ndx
    ncy = pc[1] + ndy

    x1 = int(round(ncx - w / 2.0))
    y1 = int(round(ncy - h / 2.0))
    x2 = int(round(ncx + w / 2.0))
    y2 = int(round(ncy + h / 2.0))
    return [x1, y1, x2, y2]


def ema_box(prev_box, cur_box, alpha):
    return [
        int(round(alpha * cur_box[i] + (1.0 - alpha) * prev_box[i]))
        for i in range(4)
    ]


def stabilize_track(rows, alpha=0.6, max_gap=3, min_score=0.0, max_center_step=40.0):
    rows = sorted(rows, key=lambda x: int(x["frame_idx"]))
    rows = [r for r in rows if float(r.get("score", 0.0)) >= min_score]
    if not rows:
        return []

    out = []
    prev = dict(rows[0])
    prev["stabilized"] = True
    prev["gap_filled"] = False
    out.append(prev)

    for cur in rows[1:]:
        gap = int(cur["frame_idx"]) - int(prev["frame_idx"])

        if 1 < gap <= max_gap:
            for k in range(1, gap):
                t = k / gap
                synth = dict(prev)
                synth["frame_idx"] = int(prev["frame_idx"]) + k
                synth["box"] = interp_box(prev["box"], cur["box"], t)
                synth["score"] = float(min(float(prev["score"]), float(cur["score"])))
                synth["stabilized"] = True
                synth["gap_filled"] = True
                out.append(synth)

        cur2 = dict(cur)
        smooth = ema_box(prev["box"], cur["box"], alpha)
        smooth = clamp_box_step(prev["box"], smooth, max_center_step=max_center_step)
        cur2["box"] = smooth
        cur2["stabilized"] = True
        cur2["gap_filled"] = False
        out.append(cur2)
        prev = cur2

    return out


def stabilize_rows(rows, alpha=0.6, max_gap=3, min_score=0.0, max_center_step=40.0):
    by_track = defaultdict(list)
    for row in rows:
        key = (str(row["track_id"]), str(row.get("label", "")))
        by_track[key].append(row)

    out = []
    for key in sorted(by_track):
        out.extend(
            stabilize_track(
                by_track[key],
                alpha=alpha,
                max_gap=max_gap,
                min_score=min_score,
                max_center_step=max_center_step,
            )
        )

    out = sorted(out, key=lambda x: (int(x["frame_idx"]), str(x["track_id"])))
    meta = {
        "input_rows": len(rows),
        "output_rows": len(out),
        "gap_filled_rows": sum(1 for r in out if r.get("gap_filled")),
        "track_groups": len(by_track),
    }
    return out, meta
