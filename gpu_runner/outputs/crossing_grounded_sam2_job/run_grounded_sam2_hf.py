import json
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
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
VIDEO_PATH = str(JOB_DIR / "input_video.mp4")
TEXT_PROMPT = "person . pedestrian . people"
OUTPUT_VIDEO_PATH = str(JOB_DIR / "raw_grounded_sam2" / "annotated.mp4")
SOURCE_VIDEO_FRAME_DIR = str(JOB_DIR / "raw_grounded_sam2" / "frames")
SAVE_TRACKING_RESULTS_DIR = str(JOB_DIR / "raw_grounded_sam2" / "annotated_frames")
ANCHOR_PATH = str(JOB_DIR / "anchors.json")
PRED_PATH = str(JOB_DIR / "preds.json")

PROMPT_TYPE_FOR_VIDEO = "box"
REGROUND_STRIDE = 10
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.3
ANN_FRAME_IDX = 0

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = r"/workspace/Grounded-SAM-2/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

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
