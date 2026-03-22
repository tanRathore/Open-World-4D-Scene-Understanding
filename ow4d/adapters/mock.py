from hashlib import md5
import math

def _seed(text):
    return int(md5(text.encode("utf-8")).hexdigest()[:8], 16)

def _box_for(label, frame_idx, width, height):
    seed = _seed(label)
    phase = (seed % 360) * math.pi / 180.0

    cx = 0.2 + ((seed % 500) / 1000.0)
    cy = 0.2 + (((seed // 7) % 500) / 1000.0)

    dx = 0.08 * math.sin(frame_idx / 9.0 + phase)
    dy = 0.06 * math.cos(frame_idx / 11.0 + phase)

    bw = max(50, int(width * (0.12 + ((seed % 90) / 1000.0))))
    bh = max(40, int(height * (0.10 + (((seed // 13) % 90) / 1000.0))))

    x1 = int((cx + dx) * width)
    y1 = int((cy + dy) * height)

    x1 = max(0, min(width - bw - 1, x1))
    y1 = max(0, min(height - bh - 1, y1))
    x2 = x1 + bw
    y2 = y1 + bh

    score = 0.55 + ((seed % 35) / 100.0)
    return [x1, y1, x2, y2], round(min(score, 0.95), 3)

def run(frames, prompt_groups, width, height):
    obs = []
    for frame in frames:
        frame_idx = frame["frame_idx"]
        for group in prompt_groups:
            box, score = _box_for(group["name"], frame_idx, width, height)
            obs.append({
                "frame_idx": frame_idx,
                "track_id": group["name"],
                "label": group["name"],
                "score": score,
                "box": box,
                "source_prompt": group["prompts"][0]
            })
    return obs
