import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

# --- Config ---
MODEL_PATH = "/home/jazzy/sem-farm/SEGMENTATION/HR-net/model/hrnet_farmland_best.pth"
INPUT_IMAGE = "/home/jazzy/sem-farm/2013.jpg"
OUTPUT_BINARY = "/home/jazzy/sem-farm/SEGMENTATION/HR-net/output/hrnet_output_opt2013.jpg"

CHUNK_SIZE = 256
OVERLAP = 64
THRESHOLD = 0.5

# --- Load model ---

def load_model():
    model = deeplabv3_resnet50(weights=None, num_classes=1, aux_loss=True)
    state_dict = torch.load(MODEL_PATH, map_location='cuda')

    # Remove incompatible aux classifier weights
    for key in list(state_dict.keys()):
        if "aux_classifier.4" in key:
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model.cuda()



# --- Split image into chunks ---
def split_image(image, chunk_size, overlap):
    height, width = image.shape[:2]
    step = chunk_size - overlap

    pad_x = (step - width % step) if width % step != 0 else 0
    pad_y = (step - height % step) if height % step != 0 else 0
    padded_img = cv2.copyMakeBorder(image, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT)

    chunks, positions = [], []

    for y in range(0, height, step):
        for x in range(0, width, step):
            x1 = max(0, x - overlap // 2)
            y1 = max(0, y - overlap // 2)
            x2 = x1 + chunk_size
            y2 = y1 + chunk_size

            chunk = padded_img[y1:y2, x1:x2]
            chunk = cv2.resize(chunk, (chunk_size, chunk_size))
            chunks.append(chunk)
            positions.append((x1, y1))

    return chunks, positions, image.shape

# --- Predict binary masks for each chunk ---
def predict_chunks(model, chunks):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    masks = []
    for chunk in tqdm(chunks, desc="Predicting"):
        input_tensor = transform(chunk).unsqueeze(0).cuda()
        with torch.no_grad():
            pred = model(input_tensor)['out']
            prob = torch.sigmoid(pred).squeeze().cpu().numpy()
            binary = (prob > THRESHOLD).astype(np.uint8)
            masks.append(binary)
    return masks

# --- Merge masks ---
def merge_masks(masks, positions, original_shape, chunk_size, overlap):
    height, width = original_shape[:2]
    full_mask = np.zeros((height, width), dtype=np.float32)
    count_mask = np.zeros((height, width), dtype=np.float32)

    weights = np.ones((chunk_size, chunk_size), dtype=np.float32)
    for i in range(overlap):
        weights[i, :] *= (i + 1) / (overlap + 1)
        weights[-i - 1, :] *= (i + 1) / (overlap + 1)
        weights[:, i] *= (i + 1) / (overlap + 1)
        weights[:, -i - 1] *= (i + 1) / (overlap + 1)

    for mask, (x, y) in zip(masks, positions):
        h, w = mask.shape
        end_x = min(x + w, width)
        end_y = min(y + h, height)
        actual_w = end_x - x
        actual_h = end_y - y

        full_mask[y:end_y, x:end_x] += mask[:actual_h, :actual_w] * weights[:actual_h, :actual_w]
        count_mask[y:end_y, x:end_x] += weights[:actual_h, :actual_w]

    merged = full_mask / np.maximum(count_mask, 1e-5)
    return (merged > 0.5).astype(np.uint8)

# --- Save mask ---
def save_binary_mask(mask, path):
    mask_img = (mask * 255).astype(np.uint8)
    cv2.imwrite(path, mask_img)

# --- Main ---
def main():
    model = load_model()
    image = cv2.imread(INPUT_IMAGE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Splitting image...")
    chunks, positions, original_shape = split_image(image, CHUNK_SIZE, OVERLAP)

    print("Running prediction...")
    masks = predict_chunks(model, chunks)

    print("Merging masks...")
    merged_mask = merge_masks(masks, positions, original_shape, CHUNK_SIZE, OVERLAP)

    print(f"Saving to {OUTPUT_BINARY}")
    save_binary_mask(merged_mask, OUTPUT_BINARY)
    print("Done!")

if __name__ == "__main__":
    main()
