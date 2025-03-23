import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from rembg import remove
import clip

train_image_path_clean = "plates/train/cleaned"
train_image_path_dirty = "plates/train/dirty"
test_image_path = "plates/test"

image_files_clean = [image for image in os.listdir(train_image_path_clean) if image.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files_dirty = [image for image in os.listdir(train_image_path_dirty) if image.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files_test = sorted([image for image in os.listdir(test_image_path) if image.lower().endswith(('.png', '.jpg', '.jpeg'))])


def remove_bg(image_path):
    with open(image_path, "rb") as f:
        img_data = f.read()
    img_no_bg = remove(img_data)
    img_np = np.frombuffer(img_no_bg, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
    
    if img.shape[-1] == 4:
        alpha_channel = img[:, :, 3]
        white_background = np.ones_like(img[:, :, :3]) * 255
        img = np.where(alpha_channel[:, :, None] > 0, img[:, :, :3], white_background)
    
    return img

IMG_SIZE = (224, 224)
train_images, y = [], []

for image_file in image_files_clean:
    img_path = os.path.join(train_image_path_clean, image_file)
    img = remove_bg(img_path)
    img = cv2.resize(img, IMG_SIZE)
    train_images.append(img)
    y.append("1")

for image_file in image_files_dirty:
    img_path = os.path.join(train_image_path_dirty, image_file)
    img = remove_bg(img_path)
    img = cv2.resize(img, IMG_SIZE)
    train_images.append(img)
    y.append("0")

print("Training images loaded and backgrounds removed successfully ✅")

test_images = []

for img in image_files_test:
    img_path = os.path.join(test_image_path, img)
    img = remove_bg(img_path)
    img = cv2.resize(img, IMG_SIZE)
    test_images.append(img)
print("Test images loaded and backgrounds removed successfully ✅")

