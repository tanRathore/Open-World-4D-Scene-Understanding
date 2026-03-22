import argparse
from pathlib import Path
import json

from schema import validate_rows, save_json

def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def convert_rows(raw):
    out = []

    if isinstance(raw, list):
        if raw and isinstance(raw[0], dict) and "objects" in raw[0]:
            for frame in raw:
                frame_idx = int(frame["frame_idx"])
                for obj in frame.get("objects", []):
                    out.append({
                        "frame_idx": frame_idx,
                        "track_id": str(obj["track_id"]),
                        "label": str(obj["label"]),
                        "score": float(obj["score"]),
                        "box": [int(v) for v in obj["box"]],
                        "source_prompt": str(obj.get("source_prompt", obj["label"]))
                    })
            return out

        if raw and isinstance(raw[0], dict) and "box" in raw[0]:
            for row in raw:
                out.append({
                    "frame_idx": int(row["frame_idx"]),
                    "track_id": str(row["track_id"]),
                    "label": str(row["label"]),
                    "score": float(row["score"]),
                    "box": [int(v) for v in row["box"]],
                    "source_prompt": str(row.get("source_prompt", row["label"]))
                })
            return out

    raise RuntimeError("bad raw preds format")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    raw = load_json(args.input)
    rows = convert_rows(raw)
    rows = validate_rows(rows)
    save_json(args.output, rows)

    print("ok")
    print(args.output)

if __name__ == "__main__":
    main()
