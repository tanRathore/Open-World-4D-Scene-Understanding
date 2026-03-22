import argparse
import json
from collections import defaultdict
from pathlib import Path


def blend_box(box_a, box_b, w):
    return [
        int(round((1.0 - w) * box_a[i] + w * box_b[i]))
        for i in range(4)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--anchor-weight", type=float, default=0.7)
    args = ap.parse_args()

    preds = json.load(open(args.preds, "r", encoding="utf-8"))
    anchors = json.load(open(args.anchors, "r", encoding="utf-8"))

    best_anchor = {}
    for a in anchors:
        key = (int(a["frame_idx"]), str(a["label"]))
        if key not in best_anchor or float(a["score"]) > float(best_anchor[key]["score"]):
            best_anchor[key] = a

    out = []
    fused = 0

    for row in preds:
        row2 = dict(row)
        key = (int(row["frame_idx"]), str(row["label"]))
        anchor = best_anchor.get(key)

        if anchor is not None:
            row2["box"] = blend_box(row["box"], anchor["box"], args.anchor_weight)
            row2["anchor_fused"] = True
            row2["anchor_score"] = float(anchor["score"])
            fused += 1
        else:
            row2["anchor_fused"] = False

        out.append(row2)

    out = sorted(out, key=lambda x: (int(x["frame_idx"]), str(x["track_id"])))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("preds ", len(preds))
    print("anchors", len(anchors))
    print("fused ", fused)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
