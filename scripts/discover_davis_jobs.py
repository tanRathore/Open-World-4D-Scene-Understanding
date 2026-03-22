from pathlib import Path
import argparse
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_json(path):
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return {}


def _sequence_name_from_job_dir(job_dir):
    name = job_dir.name
    suffix = "_grounded_sam2_job"
    if name.endswith(suffix):
        return name[:-len(suffix)]
    return name


def _prompt_groups_to_text(prompt_groups):
    parts = []
    for g in prompt_groups or []:
        name = str(g.get("name", "")).strip()
        prompts = [str(x).strip() for x in (g.get("prompts", []) or []) if str(x).strip()]
        if not name:
            continue
        if prompts:
            parts.append(f"{name}=" + ",".join(prompts))
        else:
            parts.append(name)
    return ";".join(parts)


def _extract_text(meta):
    for key in ["text", "prompt_text", "text_prompt", "prompt"]:
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    for key in ["prompt_groups", "prompts"]:
        val = meta.get(key)
        if isinstance(val, list):
            text = _prompt_groups_to_text(val)
            if text:
                return text

    nested = meta.get("job", {})
    if isinstance(nested, dict):
        text = _extract_text(nested)
        if text:
            return text

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-root", default="gpu_runner/outputs")
    ap.add_argument("--data-root", default="data/raw/davis/DAVIS/JPEGImages/480p")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gpu_root = Path(args.gpu_root)
    data_root = Path(args.data_root)

    rows = []

    for job_dir in sorted(gpu_root.glob("*_grounded_sam2_job")):
        if not job_dir.is_dir():
            continue

        seq = _sequence_name_from_job_dir(job_dir)
        preds_path = job_dir / "preds.json"
        if not preds_path.exists():
            continue

        input_path = data_root / seq
        if not input_path.exists():
            continue

        meta = {}
        for name in ["job.json", "prepared_job.json"]:
            p = job_dir / name
            if p.exists():
                meta.update(_read_json(p))

        text = _extract_text(meta) or f"{seq}={seq}"

        row = {
            "name": seq,
            "input": str(input_path),
            "text": text,
            "pred_path": str(preds_path),
        }

        anchor_path = job_dir / "anchors.json"
        if anchor_path.exists():
            row["anchor_path"] = str(anchor_path)

        rows.append(row)

    out = {
        "sequence_count": len(rows),
        "sequences": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("wrote", out_path)
    print("sequence_count", len(rows))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
