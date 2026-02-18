import torch
import torch.nn as nn
import torch.nn.functional as F

class SER_CNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SER_CNN, self).__init__()
        
        # --- Block 1: Low-Level Features ---
        # Input: (1, 128, 130) -> Output: (16, 64, 65)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.1)

        # --- Block 2: Shape Features ---
        # Input: (16, 64, 65) -> Output: (32, 32, 32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.1)

        # --- Block 3: Texture Features ---
        # Input: (32, 32, 32) -> Output: (64, 16, 16)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.1)

        # --- Block 4: Abstract Features ---
        # Input: (64, 16, 16) -> Output: (128, 8, 8)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.1)

        # --- Classifier ---
        # Flatten: 128 channels * 8 height * 8 width
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)

        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x