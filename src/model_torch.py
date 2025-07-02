import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 54 * 54, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 111, 111]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 54, 54]
        x = x.view(-1, 64 * 54 * 54)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # خروجی باینری بین 0 تا 1
        return x
