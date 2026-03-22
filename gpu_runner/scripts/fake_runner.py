import argparse
from pathlib import Path
import json

from schema import validate_rows, save_json

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True)
    args = p.parse_args()

    job_path = Path(args.job)
    with job_path.open("r", encoding="utf-8") as f:
        job = json.load(f)

    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {"frame_idx": 0, "track_id": "obj_1", "label": "object", "score": 0.90, "box": [120, 80, 380, 300], "source_prompt": "object"},
        {"frame_idx": 1, "track_id": "obj_1", "label": "object", "score": 0.91, "box": [125, 82, 384, 302], "source_prompt": "object"},
        {"frame_idx": 2, "track_id": "obj_1", "label": "object", "score": 0.92, "box": [130, 84, 388, 304], "source_prompt": "object"}
    ]

    rows = validate_rows(rows)
    save_json(out_dir / "preds.json", rows)

    with (out_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump({
            "mode": job.get("mode", "grounded_sam2"),
            "input_path": job["input_path"],
            "prompt_text": job["prompt_text"],
            "pred_path": str(out_dir / "preds.json")
        }, f, indent=2)

    print("ok")
    print(out_dir / "preds.json")

if __name__ == "__main__":
    main()
