import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt

# =======================
# Configuration
# =======================
DATA_DIR = "/home/jazzy/sem-farm/datasets"
TRAIN_ANNOTATION = os.path.join(DATA_DIR, "train", "_annotations.coco.json")
VALID_ANNOTATION = os.path.join(DATA_DIR, "valid", "_annotations.coco.json")
MODEL_SAVE_PATH = "farm_class_bestnew60.pth"
CLASS_COLORS = {
    1: (255, 0, 0),
    2: (0, 255, 0),
    
}
IMG_SIZE = (256, 256)
NUM_CLASSES = 4
BATCH_SIZE = 4
NUM_EPOCHS = 60
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# Attention Block
# =======================
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

# =======================
# AM-UNet
# =======================
class AMUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
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

# =======================
# Dice Loss
# =======================
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.softmax(inputs, dim=1)
        targets = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

# =======================
# Dataset
# =======================
class RoadDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annotation_file, transform=None):
        self.img_dir = img_dir
        self.coco = COCO(annotation_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)

        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            category_id = ann['category_id']
            mask = np.maximum(mask, (self.coco.annToMask(ann) * category_id))

        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(mask, dtype=torch.long)

# =======================
# Transforms
# =======================
transform = transforms.Compose([
    transforms.ToTensor()
])

# =======================
# Data Loaders
# =======================
train_dataset = RoadDataset(os.path.join(DATA_DIR, "train"), TRAIN_ANNOTATION, transform)
valid_dataset = RoadDataset(os.path.join(DATA_DIR, "valid"), VALID_ANNOTATION, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# =======================
# Model Setup
# =======================
model = AMUNet(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
criterion = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
scaler = torch.cuda.amp.GradScaler()

# =======================
# Training Loop
# =======================
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(DEVICE, non_blocking=True), masks.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to(DEVICE, non_blocking=True), masks.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

    scheduler.step()
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(valid_loader):.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

# =======================
# Prediction + Visualization
# =======================
def predict_and_visualize(img_path, model, class_colors):
    model.eval()
    img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor).argmax(1).squeeze().cpu().numpy()

    mask = np.zeros((*output.shape, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        mask[output == class_id] = color

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()
