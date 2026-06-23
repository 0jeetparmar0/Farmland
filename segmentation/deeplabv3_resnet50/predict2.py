import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

# Configuration
MODEL_PATH = "/home/jazzy/sem-farm/SEGMENTATION/Deeplabv3+/model/best_model150.pth"
INPUT_IMAGE = "/home/jazzy/sem-farm/data/final_farmland_jpg.jpg"
OUTPUT_BINARY = "/home/jazzy/sem-farm/SEGMENTATION/Deeplabv3+/output/150output.jpg"
CHUNK_SIZE = 256
OVERLAP = 64
THRESHOLD = 0.5
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
def load_model():
    model = deeplabv3_resnet50(weights=None, aux_loss=False)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("aux_classifier")}
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(DEVICE)
    return model

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split image into overlapping chunks with zero-padding
def split_image(image, chunk_size, overlap):
    h, w, _ = image.shape
    step = chunk_size - overlap

    chunks, positions = [], []

    for y in range(0, h, step):
        for x in range(0, w, step):
            chunk = image[y:y+chunk_size, x:x+chunk_size]

            # Pad if needed (bottom or right edge)
            pad_y = max(0, chunk_size - chunk.shape[0])
            pad_x = max(0, chunk_size - chunk.shape[1])
            if pad_y > 0 or pad_x > 0:
                chunk = cv2.copyMakeBorder(chunk, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            chunks.append(chunk)
            positions.append((x, y))

    padded_height = ((h - 1) // step + 1) * step + overlap
    padded_width = ((w - 1) // step + 1) * step + overlap
    return chunks, positions, (padded_height, padded_width)


# Batched prediction
def predict_chunks(model, chunks):
    masks = []
    with torch.no_grad():
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i:i+BATCH_SIZE]
            imgs = [transform(Image.fromarray(c).convert("RGB")) for c in batch_chunks]
            inputs = torch.stack(imgs).to(DEVICE)
            outputs = model(inputs)["out"]
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            masks.extend((probs > THRESHOLD).astype(np.uint8))
    return masks

# Merge predicted masks
def merge_masks(masks, positions, full_shape, chunk_size, overlap):
    h, w = full_shape
    step = chunk_size - overlap
    full_mask = np.zeros((h, w), dtype=np.float32)
    weight_mask = np.zeros((h, w), dtype=np.float32)

    # Smooth weights for blending
    w_map = np.ones((chunk_size, chunk_size), dtype=np.float32)
    for i in range(overlap):
        decay = (i + 1) / (overlap + 1)
        w_map[i, :] *= decay
        w_map[-(i + 1), :] *= decay
        w_map[:, i] *= decay
        w_map[:, -(i + 1)] *= decay

    for mask, (x, y) in zip(masks, positions):
        x_end, y_end = x + chunk_size, y + chunk_size
        full_mask[y:y_end, x:x_end] += mask * w_map
        weight_mask[y:y_end, x:x_end] += w_map

    final = (full_mask / (weight_mask + 1e-8)) > 0.5
    return final.astype(np.uint8)

# Save result
def save_binary_mask(mask, path):
    cv2.imwrite(path, (mask * 255).astype(np.uint8))

# Main
def main():
    print("Loading model...")
    model = load_model()

    print("Reading image...")
    image = cv2.imread(INPUT_IMAGE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Splitting image...")
    chunks, positions, full_shape = split_image(image, CHUNK_SIZE, OVERLAP)

    print(f"Predicting {len(chunks)} chunks...")
    masks = predict_chunks(model, chunks)

    print("Merging masks...")
    final_mask = merge_masks(masks, positions, full_shape, CHUNK_SIZE, OVERLAP)

    print(f"Saving output to {OUTPUT_BINARY}")
    save_binary_mask(final_mask, OUTPUT_BINARY)

    print("✅ Prediction complete.")

if __name__ == "__main__":
    main()
