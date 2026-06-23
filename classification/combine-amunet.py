import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms

# ========================
# Configuration
# ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/jazzy/sem-farm/farm_class_bestnew60.pth"
IMG_SIZE = (256, 256)
NUM_CLASSES = 4
CLASS_COLORS = {
    1: (255, 0, 0),   # Red
    2: (0, 255, 0),   # Green
}

transform = transforms.Compose([
    transforms.ToTensor()
])

# ========================
# Attention Block
# ========================
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# ========================
# AM-UNet Definition
# ========================
class AMUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=NUM_CLASSES):
        super(AMUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(128, 128, 64)
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(64, 64, 32)
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)

        d1 = self.up1(e3)
        a1 = self.att1(d1, e2)
        d1 = torch.cat((a1, d1), dim=1)
        d1 = self.up_conv1(d1)

        d2 = self.up2(d1)
        a2 = self.att2(d2, e1)
        d2 = torch.cat((a2, d2), dim=1)
        d2 = self.up_conv2(d2)

        out = self.final_conv(d2)
        return out

# ========================
# Helper: Colorize Mask
# ========================
def colorize_mask(mask):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask

# ========================
# Helper: Predict on Chunk
# ========================
def predict_chunk(chunk, model):
    chunk_tensor = transform(chunk).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(chunk_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return pred

# ========================
# Main Prediction Function
# ========================
def predict_on_large_image(img_path, model, save_path=None):
    original_img = Image.open(img_path).convert('RGB')
    img_np = np.array(original_img)
    H, W = img_np.shape[:2]
    tile_h, tile_w = IMG_SIZE
    final_mask = np.zeros((H, W), dtype=np.uint8)

    for y in range(0, H, tile_h):
        for x in range(0, W, tile_w):
            y1, y2 = y, min(H, y + tile_h)
            x1, x2 = x, min(W, x + tile_w)

            chunk = img_np[y1:y2, x1:x2]
            padded_chunk = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            padded_chunk[:y2 - y1, :x2 - x1] = chunk
            pil_chunk = Image.fromarray(padded_chunk)

            pred_chunk = predict_chunk(pil_chunk, model)
            final_mask[y1:y2, x1:x2] = pred_chunk[:y2 - y1, :x2 - x1]

    colored_mask = colorize_mask(final_mask)

    if save_path:
        Image.fromarray(colored_mask).save(save_path)

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(colored_mask)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ========================
# Run Prediction
# ========================
if __name__ == "__main__":
    image_path = "/home/jazzy/sem-farm/data/final_farmland_jpg.jpg"  # Replace with your test image path
    output_path = "/home/jazzy/sem-farm/CLASSIFICATION/AM-UNET/outputnew60.png"

    model = AMUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    predict_on_large_image(image_path, model, save_path=output_path)
