import argparse
from pathlib import Path
import json
import yaml

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="gpu_runner/configs/base.yaml")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg["input_path"] = args.input
    cfg["prompt_text"] = args.text
    cfg["out_dir"] = args.out

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "job.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("ok")
    print(out / "job.json")

if __name__ == "__main__":
    main()
