from pathlib import Path
import json
import subprocess
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]


def tail(text, n=20):
    lines = (text or "").splitlines()
    return "\n".join(lines[-n:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = json.load(open(args.spec, "r", encoding="utf-8"))
    rows = []

    for item in spec["sequences"]:
        seq = item["name"]
        text = item["text"]

        cmd = [
            sys.executable,
            "gpu_runner/scripts/make_davis_grounded_job.py",
            "--sequence",
            seq,
            "--text",
            text,
        ]

        print("run", seq)
        res = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

        row = {
            "sequence": seq,
            "text": text,
            "ok": res.returncode == 0,
            "returncode": res.returncode,
            "stdout_tail": tail(res.stdout),
            "stderr_tail": tail(res.stderr),
            "job_dir": f"gpu_runner/outputs/{seq}_grounded_sam2_job",
            "tar_path": f"gpu_runner/outputs/{seq}_grounded_sam2_job.tar.gz",
        }
        rows.append(row)

        print("ok" if row["ok"] else "fail", seq)
        if row["stdout_tail"]:
            print(row["stdout_tail"])
        if row["stderr_tail"]:
            print(row["stderr_tail"])

    out = {
        "spec": args.spec,
        "count": len(rows),
        "ok_count": sum(1 for r in rows if r["ok"]),
        "fail_count": sum(1 for r in rows if not r["ok"]),
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("wrote", out_path)
    print("ok_count", out["ok_count"])
    print("fail_count", out["fail_count"])


if __name__ == "__main__":
    main()
