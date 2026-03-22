from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path

from ow4d.prompts import parse_prompt_groups
from ow4d.stabilization import (
    apply_windowed_anchor_fusion,
    canonicalize_anchor_rows,
    canonicalize_rows,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--base-weight", type=float, default=0.5)
    args = ap.parse_args()

    preds = json.load(open(args.preds, "r", encoding="utf-8"))
    anchors = json.load(open(args.anchors, "r", encoding="utf-8"))
    prompt_groups = parse_prompt_groups(args.text) if args.text else []

    pred_canon = {"rows": len(preds), "changed": 0}
    anchor_canon = {"rows": len(anchors), "changed": 0}

    if prompt_groups:
        preds, pred_canon = canonicalize_rows(preds, prompt_groups)
        anchors, anchor_canon = canonicalize_anchor_rows(anchors, prompt_groups)

    out, fuse_meta = apply_windowed_anchor_fusion(
        preds,
        anchors,
        window=args.window,
        base_weight=args.base_weight,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("pred_canon", pred_canon)
    print("anchor_canon", anchor_canon)
    print("anchor_fusion", fuse_meta)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
