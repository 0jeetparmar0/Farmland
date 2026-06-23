import os
import cv2
import torch
import json
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from pycocotools.coco import COCO
from torchvision.transforms import functional as TF

# --- Config ---
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 150
LR = 1e-4
THRESHOLD = 0.5

# --- Paths ---
DATASET_PATH = "/home/jazzy/sem-farm/Spatial_temporal/2025_data"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VALID_DIR = os.path.join(DATASET_PATH, "valid")
TEST_DIR = os.path.join(DATASET_PATH, "test")
ANNOT_PATHS = {
    "train": os.path.join(TRAIN_DIR, "_annotations.coco.json"),
    "valid": os.path.join(VALID_DIR, "_annotations.coco.json"),
    "test": os.path.join(TEST_DIR, "_annotations.coco.json"),
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Dataset Class ---
class FarmlandDataset(Dataset):
    def __init__(self, image_dir, annotation_path, transforms=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        self.image_ids = self.coco.getImgIds()
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])

        image = Image.open(img_path).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        for ann in anns:
            mask += self.coco.annToMask(ann)

        mask = Image.fromarray(mask).resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
        mask = np.array(mask) > 0

        if self.transforms:
            image = self.transforms(image)

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask


# --- Transforms ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# --- Load Datasets ---
train_ds = FarmlandDataset(TRAIN_DIR, ANNOT_PATHS["train"], transforms=transform)
valid_ds = FarmlandDataset(VALID_DIR, ANNOT_PATHS["valid"], transforms=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False)


# --- Model ---
model = deeplabv3_resnet50(pretrained=True)
model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
model.to(DEVICE)


# --- Loss and Optimizer ---
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# --- Train Loop ---
def train():
    best_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)['out']
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            torch.save(model.state_dict(), "/home/jazzy/sem-farm/SEGMENTATION/Deeplabv3+/model/2025_model150.pth")
            best_loss = avg_loss


# --- Inference / Save Sample Masks ---
def save_predictions(loader, save_dir="predictions"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx, (img, mask) in enumerate(loader):
            img = img.to(DEVICE)
            output = model(img)['out']
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask = (pred > THRESHOLD).astype(np.uint8) * 255

            pred_img = Image.fromarray(pred_mask)
            pred_img.save(os.path.join(save_dir, f"mask_{idx}.png"))


if __name__ == "__main__":
    train()
    save_predictions(valid_loader, save_dir="val_predictions")
