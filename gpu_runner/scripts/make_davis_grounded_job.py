import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--text", required=True, help='example: "bear=bear,animal"')
    ap.add_argument("--name", default=None, help="defaults to sequence")
    ap.add_argument("--hf-model-id", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--sam2-checkpoint", default="/workspace/Grounded-SAM-2/sam2.1_hiera_large.pt")
    ap.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--box-threshold", type=float, default=0.4)
    ap.add_argument("--text-threshold", type=float, default=0.3)
    ap.add_argument("--prompt-type-for-video", default="box")
    ap.add_argument("--ann-frame-idx", type=int, default=0)
    ap.add_argument("--reground-stride", type=int, default=10)
    args = ap.parse_args()

    root = Path(".")
    name = args.name or args.sequence
    out_dir = root / "gpu_runner" / "outputs" / f"{name}_grounded_sam2_job"
    out_dir.mkdir(parents=True, exist_ok=True)

    job = {
        "mode": "grounded_sam2",
        "input_path": f"data/raw/davis/DAVIS/JPEGImages/480p/{args.sequence}",
        "prompt_text": args.text,
        "out_dir": str(out_dir),
        "device": "cuda",
        "hf_model_id": args.hf_model_id,
        "sam2_checkpoint": args.sam2_checkpoint,
        "sam2_model_cfg": args.sam2_model_cfg,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "prompt_type_for_video": args.prompt_type_for_video,
        "ann_frame_idx": args.ann_frame_idx,
        "reground_stride": args.reground_stride,
    }

    job_path = out_dir / "job.json"
    with job_path.open("w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)

    print("wrote", job_path)

    subprocess.check_call(
        [sys.executable, "gpu_runner/scripts/grounded_sam2_runner.py", "--job", str(job_path)]
    )
    subprocess.check_call(
        [sys.executable, "gpu_runner/scripts/package_job.py", "--job-dir", str(out_dir)]
    )

    print("ready", out_dir)
    print("tar", f"{out_dir}.tar.gz")


if __name__ == "__main__":
    main()
