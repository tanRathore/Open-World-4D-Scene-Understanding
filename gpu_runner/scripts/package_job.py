import argparse
from pathlib import Path
import tarfile

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job-dir", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    job_dir = Path(args.job_dir)
    if not job_dir.exists():
        raise RuntimeError(f"missing job dir: {job_dir}")

    out = Path(args.out) if args.out else job_dir.with_suffix(".tar.gz")
    out.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(out, "w:gz") as tar:
        tar.add(job_dir, arcname=job_dir.name)

    print("ok")
    print(out)

if __name__ == "__main__":
    main()
