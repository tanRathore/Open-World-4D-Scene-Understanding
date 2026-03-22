from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_davis_sequences(root):
    root = Path(root)
    img_root = root / "JPEGImages" / "480p"
    if not img_root.exists():
        raise RuntimeError(f"bad davis root: {root}")

    rows = []
    for seq_dir in sorted(img_root.iterdir()):
        if not seq_dir.is_dir():
            continue

        n = 0
        for p in seq_dir.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                n += 1

        rows.append({
            "name": seq_dir.name,
            "frames": n,
            "path": str(seq_dir)
        })

    return rows
