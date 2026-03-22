from pathlib import Path
import json

import numpy as np
from PIL import Image


IMG_EXTS = [".png", ".jpg", ".jpeg", ".npy"]


def _stem_index(name):
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else None


def find_depth_files(depth_dir):
    if not depth_dir:
        return {}

    root = Path(depth_dir)
    if not root.exists():
        return {}

    out = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue

        idx = _stem_index(p.stem)
        if idx is None:
            continue
        if idx not in out:
            out[idx] = str(p)

    return out


def load_intrinsics(intrinsics_path):
    if not intrinsics_path:
        return None

    p = Path(intrinsics_path)
    if not p.exists():
        return None

    if p.suffix.lower() != ".json":
        return None

    data = json.load(open(p, "r", encoding="utf-8"))

    fx = data.get("fx")
    fy = data.get("fy")
    cx = data.get("cx")
    cy = data.get("cy")
    if None not in [fx, fy, cx, cy]:
        return {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "source": str(p),
        }

    k = data.get("K") or data.get("intrinsics")
    if isinstance(k, list) and len(k) == 9:
        return {
            "fx": float(k[0]),
            "fy": float(k[4]),
            "cx": float(k[2]),
            "cy": float(k[5]),
            "source": str(p),
        }

    return None


def load_depth_array(depth_path):
    p = Path(depth_path)
    if p.suffix.lower() == ".npy":
        arr = np.load(p)
    else:
        arr = np.array(Image.open(p))

    if arr.ndim == 3:
        arr = arr[..., 0]

    return arr.astype(float)


def _clean_depth_values(vals):
    vals = np.asarray(vals, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    return vals


def _clip_box_xyxy(x0, y0, x1, y1, w, h):
    xa = int(np.floor(min(float(x0), float(x1))))
    xb = int(np.ceil(max(float(x0), float(x1))))
    ya = int(np.floor(min(float(y0), float(y1))))
    yb = int(np.ceil(max(float(y0), float(y1))))

    xa = min(max(xa, 0), w - 1)
    xb = min(max(xb, 0), w - 1)
    ya = min(max(ya, 0), h - 1)
    yb = min(max(yb, 0), h - 1)

    if xb < xa:
        xa, xb = xb, xa
    if yb < ya:
        ya, yb = yb, ya

    if xb == xa:
        if xb < w - 1:
            xb += 1
        elif xa > 0:
            xa -= 1

    if yb == ya:
        if yb < h - 1:
            yb += 1
        elif ya > 0:
            ya -= 1

    return xa, ya, xb, yb


def sample_depth_at_xy(depth_path, cx, cy, patch_radius=1):
    arr = load_depth_array(depth_path)
    h, w = arr.shape[:2]

    x = int(round(float(cx)))
    y = int(round(float(cy)))

    x = min(max(x, 0), w - 1)
    y = min(max(y, 0), h - 1)

    x0 = max(0, x - patch_radius)
    x1 = min(w - 1, x + patch_radius)
    y0 = max(0, y - patch_radius)
    y1 = min(h - 1, y + patch_radius)

    vals = _clean_depth_values(arr[y0:y1 + 1, x0:x1 + 1])
    if vals.size == 0:
        return None

    return float(np.median(vals))


def sample_depth_in_box(depth_path, x0, y0, x1, y1, inner_frac=0.6, min_pixels=9, return_stats=False):
    arr = load_depth_array(depth_path)
    h, w = arr.shape[:2]

    x0, y0, x1, y1 = _clip_box_xyxy(x0, y0, x1, y1, w, h)
    rx0, ry0, rx1, ry1 = x0, y0, x1, y1

    if 0 < float(inner_frac) < 1:
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        half_w = max(((x1 - x0 + 1) * float(inner_frac)) / 2.0, 1.0)
        half_h = max(((y1 - y0 + 1) * float(inner_frac)) / 2.0, 1.0)
        rx0, ry0, rx1, ry1 = _clip_box_xyxy(
            cx - half_w,
            cy - half_h,
            cx + half_w,
            cy + half_h,
            w,
            h,
        )

    vals = _clean_depth_values(arr[ry0:ry1 + 1, rx0:rx1 + 1])

    if vals.size < int(min_pixels) and (rx0, ry0, rx1, ry1) != (x0, y0, x1, y1):
        rx0, ry0, rx1, ry1 = x0, y0, x1, y1
        vals = _clean_depth_values(arr[ry0:ry1 + 1, rx0:rx1 + 1])

    if vals.size == 0:
        stats = {
            "box_xyxy": [x0, y0, x1, y1],
            "sample_box_xyxy": [rx0, ry0, rx1, ry1],
            "value_count": 0,
            "median": None,
            "p10": None,
            "p90": None,
            "depth_span": None,
        }
        return (None, stats) if return_stats else None

    p10 = float(np.percentile(vals, 10))
    p90 = float(np.percentile(vals, 90))
    median = float(np.median(vals))

    stats = {
        "box_xyxy": [x0, y0, x1, y1],
        "sample_box_xyxy": [rx0, ry0, rx1, ry1],
        "value_count": int(vals.size),
        "median": round(median, 6),
        "p10": round(p10, 6),
        "p90": round(p90, 6),
        "depth_span": round(max(p90 - p10, 0.0), 6),
    }

    return (median, stats) if return_stats else median


def backproject_xy_depth(cx, cy, depth, intrinsics):
    if depth is None or intrinsics is None:
        return None

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    c0x = float(intrinsics["cx"])
    c0y = float(intrinsics["cy"])

    x = (float(cx) - c0x) * float(depth) / fx
    y = (float(cy) - c0y) * float(depth) / fy
    z = float(depth)

    return [round(x, 6), round(y, 6), round(z, 6)]


def build_depth_match_table(object_state_4d, depth_dir=None, intrinsics_path=None):
    depth_files = find_depth_files(depth_dir)
    intrinsics = load_intrinsics(intrinsics_path)

    rows = []
    matched = 0

    for obj in object_state_4d.get("objects", []):
        for s in obj.get("states", []):
            frame_idx = int(s["frame_idx"])
            depth_path = depth_files.get(frame_idx)
            if depth_path:
                matched += 1

            rows.append({
                "track_id": obj["track_id"],
                "label": obj["label"],
                "frame_idx": frame_idx,
                "depth_path": depth_path,
                "has_depth": depth_path is not None,
                "has_intrinsics": intrinsics is not None,
            })

    return {
        "depth_dir": depth_dir,
        "intrinsics_path": intrinsics_path,
        "depth_file_count": len(depth_files),
        "intrinsics": intrinsics,
        "matched_state_count": matched,
        "rows": rows,
    }
