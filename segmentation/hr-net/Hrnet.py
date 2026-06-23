import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models.segmentation import deeplabv3_resnet50
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
import cv2
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# --- Config ---
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 100
LR = 1e-4
THRESHOLD = 0.5

# --- Paths ---
DATASET_PATH = "/home/jazzy/sem-farm/datasets"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VALID_DIR = os.path.join(DATASET_PATH, "valid")
ANNOT_PATHS = {
    "train": os.path.join(TRAIN_DIR, "_annotations.coco.json"),
    "valid": os.path.join(VALID_DIR, "_annotations.coco.json"),
}

# --- Dataset ---
class FarmlandDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        path = self.coco.imgs[image_id]['file_name']
        image = cv2.imread(os.path.join(self.image_dir, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for ann in anns:
            cat_id = ann['category_id']
            rle = self.coco.annToMask(ann)
            mask = np.maximum(mask, rle.astype(np.uint8))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)

        return image, mask

    def __len__(self):
        return len(self.image_ids)

# --- Transform ---
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2(),
])

train_dataset = FarmlandDataset(TRAIN_DIR, ANNOT_PATHS['train'], transform)
valid_dataset = FarmlandDataset(VALID_DIR, ANNOT_PATHS['valid'], transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

# --- Model ---
model = deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)  # binary class
model = model.cuda()

# --- Loss & Optimizer ---
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- Training Loop ---
def train_model():
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

        for images, masks in loop:
            images, masks = images.cuda(), masks.float().cuda()

            preds = model(images)['out']
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        val_loss = validate_model()
        print(f"Validation Loss: {val_loss:.4f}")

        # Save model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), '/home/jazzy/sem-farm/HR-net/model/hrnet_farmland_best.pth')
            print("Model saved!")

# --- Validation ---
def validate_model():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.cuda(), masks.float().cuda()
            preds = model(images)['out']
            loss = criterion(preds, masks)
            total_loss += loss.item()
    return total_loss / len(valid_loader)

# --- Run Training ---
if __name__ == '__main__':
    train_model()
