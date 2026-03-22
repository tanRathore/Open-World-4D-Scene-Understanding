from collections import defaultdict
import math

from .labels import canonicalize_rows


def canonicalize_anchor_rows(rows, prompt_groups):
    return canonicalize_rows(rows, prompt_groups)


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def normalized_center_dist(box_a, box_b):
    ca = center(box_a)
    cb = center(box_b)
    dx = cb[0] - ca[0]
    dy = cb[1] - ca[1]
    d = math.sqrt(dx * dx + dy * dy)

    aw = max(1.0, box_a[2] - box_a[0])
    ah = max(1.0, box_a[3] - box_a[1])
    bw = max(1.0, box_b[2] - box_b[0])
    bh = max(1.0, box_b[3] - box_b[1])

    scale = max(aw, ah, bw, bh, 1.0)
    return d / scale


def center_step(box_a, box_b):
    ca = center(box_a)
    cb = center(box_b)
    dx = cb[0] - ca[0]
    dy = cb[1] - ca[1]
    return math.sqrt(dx * dx + dy * dy)


def size_step(box_a, box_b):
    aw = box_a[2] - box_a[0]
    ah = box_a[3] - box_a[1]
    bw = box_b[2] - box_b[0]
    bh = box_b[3] - box_b[1]
    dw = bw - aw
    dh = bh - ah
    return math.sqrt(dw * dw + dh * dh)


def blend_box(box_a, box_b, w):
    return [
        int(round((1.0 - w) * box_a[i] + w * box_b[i]))
        for i in range(4)
    ]


def best_anchor_for_row(row, anchors_by_label, window, min_iou, max_center_frac):
    label = str(row["label"])
    frame_idx = int(row["frame_idx"])
    candidates = anchors_by_label.get(label, [])
    best = None
    best_key = None

    for a in candidates:
        af = int(a["frame_idx"])
        d = abs(af - frame_idx)
        if d > window:
            continue

        iou = box_iou(row["box"], a["box"])
        cdist = normalized_center_dist(row["box"], a["box"])

        if iou < min_iou and cdist > max_center_frac:
            continue

        key = (d, -iou, cdist, -float(a["score"]))
        if best is None or key < best_key:
            best = dict(a)
            best["compat_iou"] = iou
            best["compat_center_frac"] = cdist
            best_key = key

    return best


def anchor_weight_for_distance(base_weight, dist, window, iou, center_frac, max_center_frac):
    if dist > window:
        return 0.0

    dist_factor = 1.0 - dist / (window + 1.0)
    geom_factor = max(iou, 1.0 - min(center_frac / max(max_center_frac, 1e-6), 1.0))
    return base_weight * dist_factor * geom_factor


def _local_temporal_cost(prev_box, cur_box, next_box):
    cost = 0.0
    if prev_box is not None:
        cost += center_step(prev_box, cur_box)
        cost += 0.35 * size_step(prev_box, cur_box)
    if next_box is not None:
        cost += center_step(cur_box, next_box)
        cost += 0.35 * size_step(cur_box, next_box)
    return cost


def should_accept_fused_box(
    prev_box,
    base_box,
    fused_box,
    next_box=None,
    center_step_factor=1.15,
    size_step_factor=1.15,
    center_slack=1.0,
    size_slack=2.0,
    local_cost_factor=1.05,
    local_cost_slack=0.5,
):
    if prev_box is None and next_box is None:
        return True

    if prev_box is not None:
        base_center = center_step(prev_box, base_box)
        fused_center = center_step(prev_box, fused_box)

        base_size = size_step(prev_box, base_box)
        fused_size = size_step(prev_box, fused_box)

        max_center = max(base_center * center_step_factor, base_center + center_slack)
        max_size = max(base_size * size_step_factor, base_size + size_slack)

        if fused_center > max_center or fused_size > max_size:
            return False

    if next_box is not None:
        base_center = center_step(base_box, next_box)
        fused_center = center_step(fused_box, next_box)

        base_size = size_step(base_box, next_box)
        fused_size = size_step(fused_box, next_box)

        max_center = max(base_center * center_step_factor, base_center + center_slack)
        max_size = max(base_size * size_step_factor, base_size + size_slack)

        if fused_center > max_center or fused_size > max_size:
            return False

    base_cost = _local_temporal_cost(prev_box, base_box, next_box)
    fused_cost = _local_temporal_cost(prev_box, fused_box, next_box)
    max_cost = max(base_cost * local_cost_factor, base_cost + local_cost_slack)

    return fused_cost <= max_cost


def apply_windowed_anchor_fusion(
    preds,
    anchors,
    window=2,
    base_weight=0.35,
    min_iou=0.25,
    max_center_frac=0.35,
):
    anchors_by_label = defaultdict(list)
    for a in anchors:
        anchors_by_label[str(a["label"])].append(a)

    by_track = defaultdict(list)
    for row in preds:
        by_track[str(row["track_id"])].append(row)

    out = []
    matched = 0
    fused = 0
    rejected = 0

    for track_id in sorted(by_track):
        rows = sorted(by_track[track_id], key=lambda x: int(x["frame_idx"]))

        for idx, row in enumerate(rows):
            row2 = dict(row)
            prev_box = out[-1]["box"] if out and str(out[-1]["track_id"]) == track_id else None
            next_box = rows[idx + 1]["box"] if idx + 1 < len(rows) else None

            anchor = best_anchor_for_row(
                row,
                anchors_by_label,
                window=window,
                min_iou=min_iou,
                max_center_frac=max_center_frac,
            )

            if anchor is not None:
                matched += 1
                dist = abs(int(anchor["frame_idx"]) - int(row["frame_idx"]))
                iou = float(anchor["compat_iou"])
                center_frac = float(anchor["compat_center_frac"])
                w = anchor_weight_for_distance(
                    base_weight=base_weight,
                    dist=dist,
                    window=window,
                    iou=iou,
                    center_frac=center_frac,
                    max_center_frac=max_center_frac,
                )

                if w > 0:
                    fused_box = blend_box(row["box"], anchor["box"], w)
                    accepted = should_accept_fused_box(
                        prev_box=prev_box,
                        base_box=row["box"],
                        fused_box=fused_box,
                        next_box=next_box,
                    )

                    if accepted:
                        row2["box"] = fused_box
                        row2["anchor_window_fused"] = True
                        row2["anchor_fusion_rejected"] = False
                        row2["anchor_frame_idx"] = int(anchor["frame_idx"])
                        row2["anchor_dist"] = dist
                        row2["anchor_weight"] = w
                        row2["anchor_score"] = float(anchor["score"])
                        row2["anchor_iou"] = iou
                        row2["anchor_center_frac"] = center_frac
                        fused += 1
                    else:
                        row2["anchor_window_fused"] = False
                        row2["anchor_fusion_rejected"] = True
                        row2["anchor_frame_idx"] = int(anchor["frame_idx"])
                        row2["anchor_dist"] = dist
                        row2["anchor_weight"] = w
                        row2["anchor_score"] = float(anchor["score"])
                        row2["anchor_iou"] = iou
                        row2["anchor_center_frac"] = center_frac
                        rejected += 1
                else:
                    row2["anchor_window_fused"] = False
                    row2["anchor_fusion_rejected"] = False
            else:
                row2["anchor_window_fused"] = False
                row2["anchor_fusion_rejected"] = False

            out.append(row2)

    out = sorted(out, key=lambda x: (int(x["frame_idx"]), str(x["track_id"])))
    meta = {
        "input_rows": len(preds),
        "anchor_rows": len(anchors),
        "matched_rows": matched,
        "fused_rows": fused,
        "rejected_rows": rejected,
    }
    return out, meta
