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
PIXEL_AREA_SQM = 0.25  # example value, adjust to your image scale

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

def overlay_mask_and_areas(image, mask, pixel_area=None):
    overlay = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    farm_areas = []

    for i, contour in enumerate(contours):
        area_pixels = cv2.contourArea(contour)
        if area_pixels < 50:
            continue

        area_sqm = area_pixels * pixel_area if pixel_area else None
        area_hectares = area_sqm / 10000 if area_sqm else None
        farm_areas.append((i + 1, area_hectares))

        # Draw and label
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = contour[0][0]

        if area_hectares:
            label = f"{area_hectares:.2f} ha"
            cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
            cv2.putText(overlay, label, (cX - 30, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return overlay, farm_areas



def save_binary_mask(mask, path):
    cv2.imwrite(path, (mask * 255).astype(np.uint8))

def save_area_list(farm_areas, output_file="area_output.txt"):
    with open(output_file, "w") as f:
        for i, ha in farm_areas:
            if ha:
                f.write(f"Farm {i}: {ha:.4f} ha\n")
        total_ha = sum(ha for _, ha in farm_areas if ha)
        f.write(f"\nTotal Farmland Area: {total_ha:.4f} hectares\n")



# --- Main ---
def main():
    print("Loading model...")
    model = load_model()

    print("Reading image...")
    image = cv2.imread(INPUT_IMAGE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Splitting image...")
    chunks, positions, original_shape = split_image(image, CHUNK_SIZE, OVERLAP)

    print(f"Predicting {len(chunks)} chunks...")
    masks = predict_chunks(model, chunks)

    print("Merging masks...")
    merged_mask = merge_masks(masks, positions, original_shape, CHUNK_SIZE, OVERLAP)

    print("Saving binary mask...")
    save_binary_mask(merged_mask, OUTPUT_BINARY)

    print("Overlaying and labeling individual farms...")
    overlay, farm_areas = overlay_mask_and_areas(image, merged_mask, PIXEL_AREA_SQM)
    overlay_output_path = OUTPUT_BINARY.replace(".jpg", "_overlay_labeled.jpg")
    cv2.imwrite(overlay_output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Labeled overlay saved to {overlay_output_path}")

    print("Saving area report...")
    save_area_list(farm_areas, OUTPUT_AREA_FILE)
    print(f"Report saved to {OUTPUT_AREA_FILE}")

if __name__ == "__main__":
    main()
