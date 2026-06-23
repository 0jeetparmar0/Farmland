import os
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
import cv2
from tensorflow.keras.models import load_model

# --- Configuration ---
IMG_SIZE = 256
TEST_DIR = "/home/jazzy/sem-farm/datasets/test"
ANNOT_PATH = os.path.join(TEST_DIR, "_annotations.coco.json")
MODEL_PATH = "/home/jazzy/unetresnet/model/unetresnet60.h5"

# --- Custom Loss Function (Required for Loading Model) ---
def dice_bce_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return 1 - dice + bce

# --- Data Loader ---
def load_data(image_dir, annotation_path):
    coco = COCO(annotation_path)
    image_ids = coco.getImgIds()
    X, Y = [], []

    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0

        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)

        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        Y.append(np.expand_dims(mask, -1))

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.uint8)

# --- Load Test Data ---
X_test, Y_test = load_data(TEST_DIR, ANNOT_PATH)

# --- Load Pretrained Model ---
model = load_model(MODEL_PATH, custom_objects={'dice_bce_loss': dice_bce_loss})

# --- Generate Predictions ---
Y_pred = model.predict(X_test, batch_size=4, verbose=1)

# --- Compute RMSE ---
# Ensure predictions are in the same range as ground truth (0 or 1)
Y_pred_binary = (Y_pred > 0.5).astype(np.float32)  # Threshold at 0.5 for binary masks
rmse = np.sqrt(np.mean((Y_test - Y_pred_binary) ** 2))  # Pixel-wise RMSE
print(f"RMSE: {rmse}")

# --- Optional: Compute RMSE per Image ---
rmse_per_image = np.sqrt(np.mean((Y_test - Y_pred_binary) ** 2, axis=(1, 2, 3)))
print(f"RMSE per image: {rmse_per_image}")
print(f"Average RMSE across images: {np.mean(rmse_per_image)}")