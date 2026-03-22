from collections import defaultdict
from hashlib import md5
from pathlib import Path
import cv2

def _color(key):
    h = md5(key.encode("utf-8")).hexdigest()
    return (
        int(h[0:2], 16),
        int(h[2:4], 16),
        int(h[4:6], 16),
    )

def _draw_box(img, item):
    x1, y1, x2, y2 = item["box"]
    label = f'{item["label"]} {item["score"]:.2f}'
    color = _color(item["track_id"])

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y0 = max(0, y1 - th - 8)
    cv2.rectangle(img, (x1, y0), (x1 + tw + 10, y1), color, -1)
    cv2.putText(
        img,
        label,
        (x1 + 5, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

def render_keyframes(frames, observations, out_path, fps=6):
    frames = sorted(frames, key=lambda x: x["frame_idx"])
    if not frames:
        return None

    by_frame = defaultdict(list)
    for obs in observations:
        by_frame[obs["frame_idx"]].append(obs)

    first = cv2.imread(frames[0]["image_path"])
    if first is None:
        raise RuntimeError("bad frame")

    h, w = first.shape[:2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for frame in frames:
        img = cv2.imread(frame["image_path"])
        if img is None:
            continue

        for item in by_frame.get(frame["frame_idx"], []):
            _draw_box(img, item)

        tag = f'frame {frame["frame_idx"]}'
        cv2.putText(
            img,
            tag,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(img)

    writer.release()
    return str(out_path)
