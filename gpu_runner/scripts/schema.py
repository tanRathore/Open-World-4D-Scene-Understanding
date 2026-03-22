from pathlib import Path
import json

REQ = ["frame_idx", "track_id", "label", "score", "box"]

def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def validate_rows(rows):
    if not isinstance(rows, list):
        raise RuntimeError("preds must be a list")

    out = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"bad row {i}")

        for k in REQ:
            if k not in row:
                raise RuntimeError(f"missing {k} in row {i}")

        box = row["box"]
        if not isinstance(box, list) or len(box) != 4:
            raise RuntimeError(f"bad box in row {i}")

        out.append({
            "frame_idx": int(row["frame_idx"]),
            "track_id": str(row["track_id"]),
            "label": str(row["label"]),
            "score": float(row["score"]),
            "box": [int(v) for v in box],
            "source_prompt": str(row.get("source_prompt", row["label"]))
        })

    out.sort(key=lambda x: (x["frame_idx"], x["track_id"]))
    return out
