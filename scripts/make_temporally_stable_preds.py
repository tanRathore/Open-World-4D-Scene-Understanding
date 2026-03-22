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
    stabilize_rows,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--anchors")
    ap.add_argument("--text", default=None)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--max-gap", type=int, default=3)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--max-center-step", type=float, default=40.0)
    ap.add_argument("--anchor-window", type=int, default=4)
    ap.add_argument("--anchor-weight", type=float, default=0.5)
    args = ap.parse_args()

    preds = json.load(open(args.preds, "r", encoding="utf-8"))
    prompt_groups = parse_prompt_groups(args.text) if args.text else []

    pred_canon = {"rows": len(preds), "changed": 0}
    if prompt_groups:
        preds, pred_canon = canonicalize_rows(preds, prompt_groups)

    stabilized, stab_meta = stabilize_rows(
        preds,
        alpha=args.alpha,
        max_gap=args.max_gap,
        min_score=args.min_score,
        max_center_step=args.max_center_step,
    )

    final_rows = stabilized
    anchor_canon = {"rows": 0, "changed": 0}
    anchor_meta = {"input_rows": len(stabilized), "anchor_rows": 0, "fused_rows": 0}

    if args.anchors:
        anchors = json.load(open(args.anchors, "r", encoding="utf-8"))
        if prompt_groups:
            anchors, anchor_canon = canonicalize_anchor_rows(anchors, prompt_groups)

        final_rows, anchor_meta = apply_windowed_anchor_fusion(
            stabilized,
            anchors,
            window=args.anchor_window,
            base_weight=args.anchor_weight,
        )

    final_rows = sorted(final_rows, key=lambda x: (int(x["frame_idx"]), str(x["track_id"])))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final_rows, f, indent=2)

    print("pred_canon", pred_canon)
    print("stab", stab_meta)
    print("anchor_canon", anchor_canon)
    print("anchor_fusion", anchor_meta)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
