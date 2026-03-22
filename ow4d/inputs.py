from pathlib import Path
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _is_video_file(path):
    return Path(path).suffix.lower() in {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

def _list_images(folder):
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files

def inspect_input(path):
    path = Path(path)

    if path.is_file() and _is_video_file(path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"bad video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps else 0.0
        cap.release()

        return {
            "input_type": "video",
            "path": str(path),
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": duration,
        }

    if path.is_dir():
        images = _list_images(path)
        if not images:
            raise RuntimeError(f"no images: {path}")

        first = cv2.imread(str(images[0]))
        if first is None:
            raise RuntimeError(f"bad image: {images[0]}")

        h, w = first.shape[:2]
        return {
            "input_type": "sequence",
            "path": str(path),
            "fps": 0.0,
            "frame_count": len(images),
            "width": w,
            "height": h,
            "duration_sec": 0.0,
        }

    raise RuntimeError(f"bad input: {path}")

def sample_frame_ids(frame_count, stride=8, max_frames=120):
    ids = list(range(0, frame_count, stride))
    return ids[:max_frames]

def export_sampled_frames(input_path, out_dir, frame_ids):
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        return _export_from_video(input_path, out_dir, frame_ids)

    if input_path.is_dir():
        return _export_from_sequence(input_path, out_dir, frame_ids)

    raise RuntimeError(f"bad input: {input_path}")

def _export_from_video(video_path, out_dir, frame_ids):
    want = set(frame_ids)
    saved = []
    cap = cv2.VideoCapture(str(video_path))
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx in want:
            out_path = out_dir / f"{idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved.append({
                "frame_idx": idx,
                "image_path": str(out_path),
                "source_idx": idx
            })

        idx += 1

    cap.release()
    return saved

def _export_from_sequence(seq_dir, out_dir, frame_ids):
    images = _list_images(seq_dir)
    saved = []

    for idx in frame_ids:
        if idx >= len(images):
            break

        src = images[idx]
        img = cv2.imread(str(src))
        if img is None:
            continue

        out_path = out_dir / f"{idx:06d}{src.suffix.lower()}"
        cv2.imwrite(str(out_path), img)
        saved.append({
            "frame_idx": idx,
            "image_path": str(out_path),
            "source_idx": idx,
            "source_path": str(src)
        })

    return saved
