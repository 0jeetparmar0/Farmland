import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models.segmentation import deeplabv3_resnet101
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# --- Config ---
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-4
THRESHOLD = 0.5
PATIENCE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Paths ---
DATASET_PATH = "/home/jazzy/sem-farm/datasets"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VALID_DIR = os.path.join(DATASET_PATH, "valid")
ANNOT_PATHS = {
    "train": os.path.join(TRAIN_DIR, "_annotations.coco.json"),
    "valid": os.path.join(VALID_DIR, "_annotations.coco.json"),
}
SAVE_PATH = '/home/jazzy/sem-farm/HR-net/model/hrnet_farmland_opt.pth'

# --- Dataset ---
class FarmlandDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        path = self.coco.imgs[image_id]['file_name']
        image = cv2.imread(os.path.join(self.image_dir, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for ann in anns:
            rle = self.coco.annToMask(ann)
            mask = np.maximum(mask, rle.astype(np.uint8))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)

        return image, mask

    def __len__(self):
        return len(self.image_ids)

# --- Transforms ---
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.RandomRotate90(),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2(),
])

# --- Dataloaders ---
train_dataset = FarmlandDataset(TRAIN_DIR, ANNOT_PATHS['train'], transform)
valid_dataset = FarmlandDataset(VALID_DIR, ANNOT_PATHS['valid'], transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=4)

# --- Model ---
model = deeplabv3_resnet101(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
model = model.to(DEVICE)

# --- Loss & Optimizer ---
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        smooth = 1.
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice + self.bce(inputs, targets)

criterion = DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

scaler = torch.cuda.amp.GradScaler()  # for AMP

# --- Metrics ---
def compute_iou(preds, masks):
    preds = torch.sigmoid(preds) > THRESHOLD
    intersection = (preds & masks.bool()).float().sum((1, 2, 3))
    union = (preds | masks.bool()).float().sum((1, 2, 3))
    return (intersection / union.clamp(min=1e-6)).mean().item()

# --- Training ---
def train_model():
    best_loss = float('inf')
    best_iou = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}]")

        for images, masks in loop:
            images, masks = images.to(DEVICE), masks.float().to(DEVICE)

            with torch.cuda.amp.autocast():
                preds = model(images)['out']
                loss = criterion(preds, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        val_loss, val_iou = validate_model()
        scheduler.step(val_loss)

        print(f"Val Loss: {val_loss:.4f} | IoU: {val_iou:.4f}")

        if val_loss < best_loss or val_iou > best_iou:
            torch.save(model.state_dict(), SAVE_PATH)
            print("✅ Model saved!")
            best_loss = min(best_loss, val_loss)
            best_iou = max(best_iou, val_iou)


# --- Validation ---
def validate_model():
    model.eval()
    val_loss = 0
    iou_score = 0
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to(DEVICE), masks.float().to(DEVICE)
            preds = model(images)['out']
            loss = criterion(preds, masks)
            val_loss += loss.item()
            iou_score += compute_iou(preds, masks)

    return val_loss / len(valid_loader), iou_score / len(valid_loader)

# --- Main ---
if __name__ == "__main__":
    start = time.time()
    train_model()
    print("Training Time:", round((time.time() - start) / 60, 2), "minutes")
