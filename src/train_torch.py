import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision import transforms
from src.model_torch import CNNModel
from sklearn.model_selection import train_test_split

class SkinDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0) / 255.0  # (1, 224, 224)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return torch.tensor(img, dtype=torch.float32), label

# جمع‌آوری داده
normal_images = [os.path.join("data/processed/normal", f) for f in os.listdir("data/processed/normal")]
abnormal_images = [os.path.join("data/processed/abnormal", f) for f in os.listdir("data/processed/abnormal")]

all_images = normal_images + abnormal_images
all_labels = [0] * len(normal_images) + [1] * len(abnormal_images)

# تقسیم داده
train_paths, val_paths, train_labels, val_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Dataset و DataLoader
train_dataset = SkinDataset(train_paths, train_labels)
val_dataset = SkinDataset(val_paths, val_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# تعریف مدل و loss و optimizer
model = CNNModel()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# آموزش
for epoch in range(10):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/10 - Loss: {total_loss:.4f}")

# ذخیره مدل
torch.save(model.state_dict(), "models/skin_ultrasound_model.pt")
print("✅ مدل ذخیره شد.")
