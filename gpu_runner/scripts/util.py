from pathlib import Path
import json
import re
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def _slug(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"

def parse_prompt_groups(text):
    groups = []
    if not text.strip():
        return groups

    for chunk in text.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue

        if "=" in chunk:
            name, raw_prompts = chunk.split("=", 1)
            name = _slug(name)
            prompts = [p.strip() for p in raw_prompts.split(",") if p.strip()]
        else:
            prompts = [p.strip() for p in chunk.split(",") if p.strip()]
            name = _slug(prompts[0]) if prompts else "item"

        if prompts:
            groups.append({
                "name": name,
                "prompts": prompts
            })

    return groups

def flat_prompts(groups):
    out = []
    for g in groups:
        out.extend(g["prompts"])
    return out

def list_images(folder):
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files

def is_video(path):
    return Path(path).suffix.lower() in {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

def sequence_to_mp4(seq_dir, out_path, fps=6):
    images = list_images(seq_dir)
    if not images:
        raise RuntimeError(f"no images: {seq_dir}")

    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"bad image: {images[0]}")

    h, w = first.shape[:2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        writer.write(img)

    writer.release()
    return str(out_path), len(images), w, h
