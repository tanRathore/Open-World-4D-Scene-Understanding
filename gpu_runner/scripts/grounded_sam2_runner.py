import argparse
from pathlib import Path

from util import load_json, save_json, parse_prompt_groups, flat_prompts, is_video, sequence_to_mp4


RUNNER_TEMPLATE = r'''import json
import os
import cv2
import torch
import numpy as np
import supervision as sv

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

JOB_DIR = Path(__file__).resolve().parent
MODEL_ID = "{model_id}"
VIDEO_PATH = str(JOB_DIR / "input_video.mp4")
TEXT_PROMPT = "{text_prompt}"
OUTPUT_VIDEO_PATH = str(JOB_DIR / "raw_grounded_sam2" / "annotated.mp4")
SOURCE_VIDEO_FRAME_DIR = str(JOB_DIR / "raw_grounded_sam2" / "frames")
SAVE_TRACKING_RESULTS_DIR = str(JOB_DIR / "raw_grounded_sam2" / "annotated_frames")
ANCHOR_PATH = str(JOB_DIR / "anchors.json")
PRED_PATH = str(JOB_DIR / "preds.json")

PROMPT_TYPE_FOR_VIDEO = "{prompt_type}"
REGROUND_STRIDE = {reground_stride}
BOX_THRESHOLD = {box_threshold}
TEXT_THRESHOLD = {text_threshold}
ANN_FRAME_IDX = {ann_frame_idx}

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = r"{sam2_checkpoint}"
model_cfg = "{model_cfg}"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_ID)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
print(video_info)

frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
source_frames.mkdir(parents=True, exist_ok=True)

with sv.ImageSink(
    target_dir_path=source_frames,
    overwrite=True,
    image_name_pattern="{:05d}.jpg"
) as sink:
    for frame in tqdm(frame_generator, desc="Saving Video Frames"):
        sink.save_image(frame)

frame_names = [
    p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

def run_grounding_on_frame(frame_idx):
    img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx])
    image = Image.open(img_path)
    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]]
    )

    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy().tolist()
    labels = results[0]["labels"]
    return image, boxes, scores, labels


anchor_rows = []
if REGROUND_STRIDE > 0:
    for frame_idx in range(0, len(frame_names), REGROUND_STRIDE):
        _, boxes, scores, labels = run_grounding_on_frame(frame_idx)
        for i, label in enumerate(labels):
            anchor_rows.append({
                "frame_idx": int(frame_idx),
                "label": str(label),
                "score": float(scores[i]),
                "box": [int(v) for v in boxes[i].tolist()],
                "source_prompt": TEXT_PROMPT,
                "anchor": True,
            })

with open(ANCHOR_PATH, "w", encoding="utf-8") as f:
    json.dump(anchor_rows, f, indent=2)

print("anchors", len(anchor_rows))
print("anchor path", ANCHOR_PATH)

inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)

ann_frame_idx = ANN_FRAME_IDX
image, input_boxes, confidences, class_names = run_grounding_on_frame(ann_frame_idx)

print(input_boxes)
print(class_names)

image_predictor.set_image(np.array(image.convert("RGB")))
OBJECTS = class_names

masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

if masks.ndim == 4:
    masks = masks.squeeze(1)

assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"]

if PROMPT_TYPE_FOR_VIDEO == "point":
    all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
    for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
        labels = np.ones((points.shape[0]), dtype=np.int32)
        video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels,
        )
elif PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )
elif PROMPT_TYPE_FOR_VIDEO == "mask":
    for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
        video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            mask=mask,
        )
else:
    raise NotImplementedError("bad prompt type")

video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

os.makedirs(SAVE_TRACKING_RESULTS_DIR, exist_ok=True)
ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

rows = []

for frame_idx, segments in video_segments.items():
    img = cv2.imread(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx]))

    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)

    xyxy = sv.mask_to_xyxy(masks)

    detections = sv.Detections(
        xyxy=xyxy,
        mask=masks,
        class_id=np.array(object_ids, dtype=np.int32),
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(
        annotated_frame,
        detections=detections,
        labels=[ID_TO_OBJECTS[i] for i in object_ids]
    )

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    cv2.imwrite(
        os.path.join(SAVE_TRACKING_RESULTS_DIR, f"annotated_frame_{frame_idx:05d}.jpg"),
        annotated_frame
    )

    for i, obj_id in enumerate(object_ids):
        score = float(confidences[obj_id - 1]) if (obj_id - 1) < len(confidences) else 1.0
        rows.append({
            "frame_idx": int(frame_idx),
            "track_id": str(obj_id),
            "label": str(ID_TO_OBJECTS[obj_id]),
            "score": score,
            "box": [int(v) for v in xyxy[i].tolist()],
            "source_prompt": TEXT_PROMPT,
        })

create_video_from_images(SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH)

with open(PRED_PATH, "w", encoding="utf-8") as f:
    json.dump(sorted(rows, key=lambda x: (x["frame_idx"], x["track_id"])), f, indent=2)

print("wrote", PRED_PATH)
'''


def make_python_runner(job, prepared):
    out_dir = Path(job["out_dir"])
    script_path = out_dir / "run_grounded_sam2_hf.py"

    text = RUNNER_TEMPLATE
    text = text.replace("{model_id}", job.get("hf_model_id", "IDEA-Research/grounding-dino-tiny"))
    text = text.replace("{text_prompt}", prepared["flat_prompt_text"])
    text = text.replace("{prompt_type}", job.get("prompt_type_for_video", "box"))
    text = text.replace("{sam2_checkpoint}", job.get("sam2_checkpoint", "/workspace/Grounded-SAM-2/sam2.1_hiera_large.pt"))
    text = text.replace("{model_cfg}", job.get("sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml"))
    text = text.replace("{reground_stride}", str(int(job.get("reground_stride", 10))))
    text = text.replace("{ann_frame_idx}", str(int(job.get("ann_frame_idx", 0))))
    text = text.replace("{box_threshold}", str(float(job.get("box_threshold", 0.4))))
    text = text.replace("{text_threshold}", str(float(job.get("text_threshold", 0.3))))

    script_path.write_text(text, encoding="utf-8")
    return str(script_path)


def make_shell(job):
    out_dir = Path(job["out_dir"])
    shell_path = out_dir / "run_grounded_sam2.sh"

    lines = [
        "#!/usr/bin/env bash",
        "set -e",
        "",
        'REPO_DIR="${REPO_DIR:-/workspace/Grounded-SAM-2}"',
        'PYTHON_BIN="${PYTHON_BIN:-python}"',
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'JOB_DIR="$SCRIPT_DIR"',
        "",
        'cd "$REPO_DIR"',
        "",
        'export PYTHONPATH="$REPO_DIR:$PYTHONPATH"',
        'echo "repo: $REPO_DIR"',
        'echo "job:  $JOB_DIR"',
        "",
        'mkdir -p "$JOB_DIR/raw_grounded_sam2"',
        "",
        '$PYTHON_BIN "$JOB_DIR/run_grounded_sam2_hf.py"',
        "",
        'echo "done"',
        'echo "$JOB_DIR/preds.json"',
    ]

    shell_path.write_text("\n".join(lines), encoding="utf-8")
    shell_path.chmod(0o755)
    return str(shell_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True)
    args = p.parse_args()

    job = load_json(args.job)
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(job["input_path"])
    prompt_groups = parse_prompt_groups(job["prompt_text"])
    flat = flat_prompts(prompt_groups)
    flat_text = " . ".join(flat)

    if input_path.is_file() and is_video(input_path):
        input_video_path = str(input_path)
        input_mode = "video"
        frame_count = None
        width = None
        height = None
    elif input_path.is_dir():
        input_video_path, frame_count, width, height = sequence_to_mp4(
            input_path,
            out_dir / "input_video.mp4",
            fps=6
        )
        input_mode = "sequence_to_video"
    else:
        raise RuntimeError(f"bad input: {input_path}")

    prepared = {
        "mode": job.get("mode", "grounded_sam2"),
        "input_mode": input_mode,
        "input_path": str(input_path),
        "input_video_path": input_video_path,
        "prompt_text": job["prompt_text"],
        "prompt_groups": prompt_groups,
        "flat_prompts": flat,
        "flat_prompt_text": flat_text,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "out_dir": str(out_dir),
    }

    save_json(out_dir / "prepared_job.json", prepared)
    py_runner = make_python_runner(job, prepared)
    shell_path = make_shell(job)

    print("ok")
    print(out_dir / "prepared_job.json")
    print(py_runner)
    print(shell_path)


if __name__ == "__main__":
    main()
