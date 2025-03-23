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

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float()

class PlateDataset(Dataset):
    def __init__(self, images, labels, preprocess):
        self.images = images
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        image = self.preprocess(image)
        label = torch.tensor(int(self.labels[idx]))
        return image, label

train_dataset = PlateDataset(train_images, y, preprocess)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class CLIPClassifier(nn.Module):
    def __init__(self):
        super(CLIPClassifier, self).__init__()
        self.clip_model = model.visual
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        with torch.no_grad():
            x = self.clip_model(x)
        return self.fc(x)

classifier = CLIPClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.fc.parameters(), lr=0.005)

num_epochs = 600
for epoch in range(num_epochs):
    classifier.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {(1- (running_loss/len(train_loader))) * 100}%")

print("Training Complete ✅")

test_dataset = PlateDataset(test_images, [0] * len(test_images), preprocess)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

classifier.eval()
predictions = []

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = classifier(images)
        predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()

        predictions.extend(["cleaned" if label == 1 else "dirty" for label in predicted_labels])
