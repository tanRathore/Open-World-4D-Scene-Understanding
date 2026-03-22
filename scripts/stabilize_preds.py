from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path

from ow4d.prompts import parse_prompt_groups
from ow4d.stabilization import canonicalize_rows, stabilize_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--max-gap", type=int, default=3)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--max-center-step", type=float, default=40.0)
    args = ap.parse_args()

    rows = json.load(open(args.input, "r", encoding="utf-8"))
    prompt_groups = parse_prompt_groups(args.text) if args.text else []

    canon_meta = {"rows": len(rows), "changed": 0}
    if prompt_groups:
        rows, canon_meta = canonicalize_rows(rows, prompt_groups)

    out, stab_meta = stabilize_rows(
        rows,
        alpha=args.alpha,
        max_gap=args.max_gap,
        min_score=args.min_score,
        max_center_step=args.max_center_step,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("canon", canon_meta)
    print("stab", stab_meta)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
