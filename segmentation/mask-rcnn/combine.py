import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo


# --------------------------
# 🔧 Configuration
# --------------------------
IMAGE_PATH = "/home/jazzy/sem-farm/data/final_farmland_jpg.jpg"
MODEL_WEIGHTS = "/home/jazzy/sem-farm/Mask-rcnn/code/output/model_final.pth"
OUTPUT_PATH = "merged_maskrcnn_binary.png"

CHUNK_SIZE = 256
OVERLAP = 64
THRESHOLD = 0.5
NUM_CLASSES = 2


# --------------------------
# 🔧 Utilities
# --------------------------
def load_image(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def split_image(image, chunk_size, overlap):
    h, w = image.shape[:2]
    step = chunk_size - overlap
    pad_x = (step - w % step) if w % step != 0 else 0
    pad_y = (step - h % step) if h % step != 0 else 0
    padded = cv2.copyMakeBorder(image, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT)

    chunks, positions = [], []
    for y in range(0, padded.shape[0] - overlap, step):
        for x in range(0, padded.shape[1] - overlap, step):
            tile = padded[y:y + chunk_size, x:x + chunk_size]
            chunks.append(tile)
            positions.append((x, y))
    return chunks, positions, image.shape


def predict_tile(predictor, tile):
    if tile.shape[2] == 4:
        tile = cv2.cvtColor(tile, cv2.COLOR_RGBA2RGB)

    outputs = predictor(tile)
    mask = np.zeros(tile.shape[:2], dtype=np.uint8)

    if "instances" in outputs:
        instances = outputs["instances"].to("cpu")
        if instances.has("pred_masks"):
            for m in instances.pred_masks.numpy():
                mask = np.maximum(mask, (m * 255).astype(np.uint8))

    return mask


def merge_masks(masks, positions, orig_shape, chunk_size, overlap):
    full_mask = np.zeros(orig_shape[:2], dtype=np.float32)
    count_mask = np.zeros_like(full_mask)

    weights = np.ones((chunk_size, chunk_size), dtype=np.float32)
    for i in range(overlap):
        weights[i, :] *= (i + 1) / (overlap + 1)
        weights[-i-1, :] *= (i + 1) / (overlap + 1)
        weights[:, i] *= (i + 1) / (overlap + 1)
        weights[:, -i-1] *= (i + 1) / (overlap + 1)

    for mask, (x, y) in zip(masks, positions):
        h, w = mask.shape
        end_x = min(x + w, orig_shape[1])
        end_y = min(y + h, orig_shape[0])
        actual_w = end_x - x
        actual_h = end_y - y

        full_mask[y:end_y, x:end_x] += mask[:actual_h, :actual_w] * weights[:actual_h, :actual_w]
        count_mask[y:end_y, x:end_x] += weights[:actual_h, :actual_w]

    final_mask = (full_mask / np.maximum(count_mask, 1e-6)) > (THRESHOLD * 255)
    return final_mask.astype(np.uint8)


def setup_cfg(weights_path, num_classes=1, score_thresh=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    return cfg


def save_binary_mask(mask, path):
    mask_img = (mask * 255).astype(np.uint8)
    cv2.imwrite(path, mask_img)


# --------------------------
# 🚀 Main Execution
# --------------------------
def main():
    print("Loading image...")
    image = load_image(IMAGE_PATH)

    print("Splitting into chunks...")
    chunks, positions, orig_shape = split_image(image, CHUNK_SIZE, OVERLAP)

    print("Setting up model...")
    cfg = setup_cfg(MODEL_WEIGHTS, NUM_CLASSES, THRESHOLD)
    predictor = DefaultPredictor(cfg)

    print("Running inference on chunks...")
    masks = []
    for tile in tqdm(chunks):
        masks.append(predict_tile(predictor, tile))

    print("Merging predictions...")
    merged = merge_masks(masks, positions, orig_shape, CHUNK_SIZE, OVERLAP)

    print(f"Saving final mask to {OUTPUT_PATH}...")
    save_binary_mask(merged, OUTPUT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
