import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

# Configuration
MODEL_PATH = "/home/jazzy/sem-farm/SEGMENTATION/Deeplabv3+/model/best_model100.pth"
INPUT_IMAGE = "/home/jazzy/sem-farm/2013.jpg"
OUTPUT_BINARY = "/home/jazzy/sem-farm/SEGMENTATION/Deeplabv3+/output/2013output.jpg"
CHUNK_SIZE = 256
OVERLAP = 64
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model

def load_model():
    model = deeplabv3_resnet50(weights=None, aux_loss=False)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
    
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Remove aux_classifier keys if present
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("aux_classifier")}
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(DEVICE)
    return model



# Image normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Split image into overlapping chunks
def split_image(image, chunk_size, overlap):
    h, w, _ = image.shape
    step = chunk_size - overlap
    pad_x = (step - w % step) if w % step != 0 else 0
    pad_y = (step - h % step) if h % step != 0 else 0
    padded_img = cv2.copyMakeBorder(image, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT)
    chunks = []
    positions = []

    for y in range(0, padded_img.shape[0] - overlap, step):
        for x in range(0, padded_img.shape[1] - overlap, step):
            chunk = padded_img[y:y + chunk_size, x:x + chunk_size]
            chunks.append(chunk)
            positions.append((x, y))
    return chunks, positions, padded_img.shape[:2]

# Predict binary masks for each chunk
def predict_chunks(model, chunks):
    masks = []
    for chunk in chunks:
        img = Image.fromarray(chunk).convert("RGB")
        img = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img)["out"]
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
            binary_mask = (prob > THRESHOLD).astype(np.uint8)
        masks.append(binary_mask)
    return masks

# Merge masks back into full image
def merge_masks(masks, positions, full_shape, chunk_size, overlap):
    h, w = full_shape
    full_mask = np.zeros((h, w), dtype=np.float32)
    count_mask = np.zeros((h, w), dtype=np.float32)

    weights = np.ones((chunk_size, chunk_size), dtype=np.float32)
    for i in range(overlap):
        weights[i, :] *= (i + 1) / (overlap + 1)
        weights[-i - 1, :] *= (i + 1) / (overlap + 1)
        weights[:, i] *= (i + 1) / (overlap + 1)
        weights[:, -i - 1] *= (i + 1) / (overlap + 1)

    for mask, (x, y) in zip(masks, positions):
        end_x = min(x + chunk_size, w)
        end_y = min(y + chunk_size, h)
        actual_w = end_x - x
        actual_h = end_y - y
        full_mask[y:end_y, x:end_x] += mask[:actual_h, :actual_w] * weights[:actual_h, :actual_w]
        count_mask[y:end_y, x:end_x] += weights[:actual_h, :actual_w]

    merged_mask = (full_mask / (count_mask + 1e-8)) > 0.5
    return merged_mask.astype(np.uint8)

# Save mask to file
def save_binary_mask(mask, output_path):
    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask)

# Main prediction pipeline
def main():
    model = load_model()
    print("Model loaded.")

    image = cv2.imread(INPUT_IMAGE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Image loaded.")

    print("Splitting image into chunks...")
    chunks, positions, full_shape = split_image(image, CHUNK_SIZE, OVERLAP)

    print(f"Predicting on {len(chunks)} chunks...")
    masks = predict_chunks(model, chunks)

    print("Merging masks...")
    merged = merge_masks(masks, positions, full_shape, CHUNK_SIZE, OVERLAP)

    print(f"Saving to {OUTPUT_BINARY}")
    save_binary_mask(merged, OUTPUT_BINARY)

    print("Prediction complete.")

if __name__ == "__main__":
    main()
