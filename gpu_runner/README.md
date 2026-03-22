# gpu_runner

This folder is for the real model side.

Goal:
- run model inference on a GPU machine
- write preds.json in the contract expected by ow4d
- keep controller and eval logic in the main repo

Main output format:
[
  {
    "frame_idx": 0,
    "track_id": "bear_1",
    "label": "bear",
    "score": 0.93,
    "box": [170, 120, 590, 430],
    "source_prompt": "bear"
  }
]
