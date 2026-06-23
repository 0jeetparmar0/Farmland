import os
import cv2
import numpy as np
import tensorflow as tf

# --- Configuration ---
MODEL_PATH = "/home/jazzy/sem-farm/SEGMENTATION/Unet_resnet50/model/unetresnetnew.h5"
INPUT_IMAGE = "/home/jazzy/sem-farm/data/test55.jpg"
OUTPUT_BINARY = "/home/jazzy/sem-farm/SEGMENTATION/Unet_resnet50/output_resnet/test5555.jpg"
OUTPUT_AREA_FILE = "/home/jazzy/sem-farm/SEGMENTATION/Unet_resnet50/output_resnet/area_output.txt"
CHUNK_SIZE = 256
OVERLAP = 64
THRESHOLD = 0.5
PIXEL_AREA_SQM = None  # Set e.g., 0.25 if known (1 pixel = 0.25 sq meters)

# --- Functions ---
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def split_image(image, chunk_size=256, overlap=32):
    height, width = image.shape[:2]
    chunks, positions = [], []
    step = chunk_size - overlap

    pad_x = (step - width % step) if width % step != 0 else 0
    pad_y = (step - height % step) if height % step != 0 else 0
    padded_img = cv2.copyMakeBorder(image, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT)

    for y in range(0, height, step):
        for x in range(0, width, step):
            x1 = max(0, x - overlap // 2)
            y1 = max(0, y - overlap // 2)
            x2 = min(width, x1 + chunk_size)
            y2 = min(height, y1 + chunk_size)
            chunk = padded_img[y1:y2, x1:x2]
            if chunk.shape[:2] != (chunk_size, chunk_size):
                chunk = cv2.resize(chunk, (chunk_size, chunk_size))
            chunks.append(chunk)
            positions.append((x1, y1))
    return chunks, positions, image.shape

def predict_chunks(model, chunks):
    masks = []
    for chunk in chunks:
        chunk = chunk / 255.0
        chunk = np.expand_dims(chunk, axis=0)
        pred = model.predict(chunk, verbose=0)[0]
        binary_mask = (pred > THRESHOLD).astype(np.uint8)
        masks.append(binary_mask)
    return masks

def merge_masks(masks, positions, original_shape, chunk_size=256, overlap=32):
    full_mask = np.zeros(original_shape[:2], dtype=np.float32)
    count_mask = np.zeros(original_shape[:2], dtype=np.float32)
    weights = np.ones((chunk_size, chunk_size), dtype=np.float32)

    for i in range(overlap):
        weights[i, :] *= (i + 1) / (overlap + 1)
        weights[-i - 1, :] *= (i + 1) / (overlap + 1)
        weights[:, i] *= (i + 1) / (overlap + 1)
        weights[:, -i - 1] *= (i + 1) / (overlap + 1)

    for mask, (x, y) in zip(masks, positions):
        h, w = mask.shape[:2]
        end_x = min(x + w, original_shape[1])
        end_y = min(y + h, original_shape[0])
        actual_w = end_x - x
        actual_h = end_y - y
        full_mask[y:end_y, x:end_x] = (
            full_mask[y:end_y, x:end_x] * count_mask[y:end_y, x:end_x] +
            mask[:actual_h, :actual_w, 0] * weights[:actual_h, :actual_w]
        ) / (count_mask[y:end_y, x:end_x] + weights[:actual_h, :actual_w])
        count_mask[y:end_y, x:end_x] += weights[:actual_h, :actual_w]

    merged_mask = (full_mask > 0.5).astype(np.uint8)
    return merged_mask

def save_binary_mask(mask, output_path):
    mask_visual = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_visual)

def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.4):
    overlay = image.copy()
    mask_indices = mask.astype(bool)
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[mask_indices] = color
    overlaid = cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0)
    return overlaid

def calculate_area(mask, pixel_area=None):
    pixel_count = np.sum(mask)
    if pixel_area:
        area_sqm = pixel_count * pixel_area
        return pixel_count, area_sqm
    return pixel_count, None

def save_area(area_pixels, area_sqm=None, output_file="area_output.txt"):
    with open(output_file, "w") as f:
        f.write(f"Farmland Area (in pixels): {area_pixels}\n")
        if area_sqm is not None:
            f.write(f"Estimated Farmland Area (in square meters): {area_sqm:.2f}\n")

# --- Main ---
def main():
    print("Loading model...")
    model = load_model()

    print("Reading image...")
    image = cv2.imread(INPUT_IMAGE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Splitting image into chunks...")
    chunks, positions, original_shape = split_image(image, CHUNK_SIZE, OVERLAP)

    print(f"Predicting {len(chunks)} chunks...")
    masks = predict_chunks(model, chunks)

    print("Merging predicted masks...")
    merged_mask = merge_masks(masks, positions, original_shape, CHUNK_SIZE, OVERLAP)

    print("Saving binary mask...")
    save_binary_mask(merged_mask, OUTPUT_BINARY)

    print("Overlaying mask on original image...")
    overlaid = overlay_mask_on_image(image, merged_mask)
    overlay_output_path = OUTPUT_BINARY.replace(".jpg", "_overlay.jpg")
    cv2.imwrite(overlay_output_path, cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
    print(f"Overlay saved to {overlay_output_path}")

    print("Calculating area...")
    pixel_count, area_sqm = calculate_area(merged_mask, PIXEL_AREA_SQM)
    save_area(pixel_count, area_sqm, OUTPUT_AREA_FILE)
    print(f"Area written to {OUTPUT_AREA_FILE}")

if __name__ == "__main__":
    main()
