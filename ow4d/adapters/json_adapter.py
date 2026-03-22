import json
from ow4d.adapters.base import Adapter, ensure_file

class JsonAdapter(Adapter):
    name = "json"

    def run(self, frames, prompt_groups, width, height):
        pred_path = self.kwargs.get("pred_path")
        if not pred_path:
            raise RuntimeError("need pred_path")

        pred_path = ensure_file(pred_path)
        with pred_path.open("r", encoding="utf-8") as f:
            rows = json.load(f)

        if not isinstance(rows, list):
            raise RuntimeError("bad json preds")

        out = []
        for row in rows:
            out.append({
                "frame_idx": int(row["frame_idx"]),
                "track_id": str(row["track_id"]),
                "label": str(row["label"]),
                "score": float(row["score"]),
                "box": [int(v) for v in row["box"]],
                "source_prompt": str(row.get("source_prompt", row["label"]))
            })

        return out
