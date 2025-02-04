import os
from itertools import product

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

import preprocessing.filters as f
from preprocessing.filters import apply_all_filters
from preprocessing.patch_generator import smash_n_reconstruct

DATA_PATH = "data/"

TRAIN_IMAGES = DATA_PATH + "train_data/"
TRAIN_IDS = DATA_PATH + "train.csv"

TEST_IMAGES = DATA_PATH + "test_data_v2/"
TEST_IDS = DATA_PATH + "test.csv"


class FeatureExtractionLayer(nn.Module):
    def __init__(self):
        super(FeatureExtractionLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # Assuming 1 input channel
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.clamp(x, min=-1, max=1)
        return x


class RichPoorTextureContrast(nn.Module):
    def __init__(self):
        super(RichPoorTextureContrast, self).__init__()

        # Feature extraction layers
        self.feature_extraction_rich = FeatureExtractionLayer()
        self.feature_extraction_poor = FeatureExtractionLayer()

        # Convolutional blocks
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv_blocks = nn.ModuleList(
            [nn.Conv2d(32, 32, kernel_size=3, padding=1) for _ in range(6)]
        )
        self.bn_blocks = nn.ModuleList([nn.BatchNorm2d(32) for _ in range(6)])

        # Average Pooling layers
        self.avgpool1 = nn.AvgPool2d(2)
        self.avgpool2 = nn.AvgPool2d(2)

        # Final layers
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, 1)

    def forward(self, input1, input2):
        # Feature extraction
        l1 = self.feature_extraction_rich(input1)
        l2 = self.feature_extraction_poor(input2)

        # Contrast layer
        contrast = l1 - l2

        # First Conv Block
        x = F.relu(self.conv1(contrast))
        x = self.bn1(x)

        # 3 Conv Blocks
        for conv, bn in zip(self.conv_blocks[:3], self.bn_blocks[:3]):
            x = F.relu(conv(x))
            x = bn(x)

        # Average Pooling Layer
        x = self.avgpool1(x)

        # 2 Conv Blocks
        for conv, bn in zip(self.conv_blocks[3:5], self.bn_blocks[3:5]):
            x = F.relu(conv(x))
            x = bn(x)

        # Average Pooling Layer
        x = self.avgpool2(x)

        # 2 Conv Blocks
        for conv, bn in zip(self.conv_blocks[5:], self.bn_blocks[5:]):
            x = F.relu(conv(x))
            x = bn(x)

        # Global Average Pooling
        x = self.global_avgpool(x)

        # Flatten and Dense layer
        x = self.flatten(x)
        x = torch.sigmoid(self.fc(x))

        return x


# Model instantiation
model = RichPoorTextureContrast()


def preprocess(path, label: int):
    # Assuming 'smash_n_reconstruct' and 'f.apply_all_filters' are available in your codebase
    rt, pt = smash_n_reconstruct(
        path
    )  # path is assumed to be a string, not a tensor here
    frt = torch.tensor(f.apply_all_filters(rt), dtype=torch.float32).unsqueeze(-1)
    fpt = torch.tensor(f.apply_all_filters(pt), dtype=torch.float32).unsqueeze(-1)
    return frt, fpt, label


def dict_map(X1, X2, y):
    return {"rich_texture": X1, "poor_texture": X2}, y


if __name__ == "__main__":
    train_csv = pd.read_csv(TRAIN_IDS)

    ai_img_paths = train_csv[train_csv.label == 1].file_name
    real_img_paths = train_csv[train_csv.label == 0].file_name

    ai_imgs = ["data/" + pth for pth in ai_img_paths]
    ai_label = [1 for i in range(len(ai_imgs))]

    real_imgs = ["data/" + pth for pth in real_img_paths]
    real_label = [0 for i in range(len(real_imgs))]

    data_ratio = int(len(ai_imgs) * 0.01)

    ai_imgs = ai_imgs[:data_ratio]
    ai_label = ai_label[:data_ratio]
    real_imgs = real_imgs[:data_ratio]
    real_label = real_label[:data_ratio]

    X_train = ai_imgs[:-21] + real_imgs[:-21]
    y_train = ai_label[:-21] + real_label[:-21]
    X_validate = ai_imgs[-21:] + real_imgs[-21:]
    y_validate = ai_label[-21:] + real_label[-21:]
    len(X_train), len(y_train), len(X_validate), len(y_validate)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    class CustomDataset(Dataset):
        def __init__(self, filepaths, labels, preprocess_fn, dict_map_fn=None):
            self.filepaths = filepaths
            self.labels = labels
            self.preprocess_fn = preprocess_fn
            self.dict_map_fn = dict_map_fn

        def __len__(self):
            return len(self.filepaths)

        def __getitem__(self, idx):
            filepath = self.filepaths[idx]
            label = self.labels[idx]
            frt, fpt, label = self.preprocess_fn(filepath, label)

            if self.dict_map_fn:
                return self.dict_map_fn(frt, fpt, label)
            return frt, fpt, label

    # Define batch size
    batch_size = 32

    # Create training and validation datasets
    train_dataset = CustomDataset(X_train, y_train, preprocess, dict_map_fn=dict_map)
    val_dataset = CustomDataset(
        X_validate, y_validate, preprocess, dict_map_fn=dict_map
    )

    # Create DataLoader for batching and prefetching
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=4)

    # The DataLoader will handle batching and shuffling, similar to TensorFlow's Dataset

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm  # Import tqdm

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )  # Use "mps" for Apple Metal devices (e.g., M1, M2)
        print("Using GPU.")
    else:
        device = torch.device("cpu")  # Fallback to CPU
        print("Using CPU.")

    # Initialize model, loss, and optimizer
    model = RichPoorTextureContrast().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    num_epochs = 10

    # Training loop with tqdm
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Wrap train loader with tqdm
        train_loop = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False
        )

        for batch_idx, batch in enumerate(train_loop, start=1):
            inputs, labels = batch
            rich = inputs["rich_texture"].permute(0, 3, 1, 2).float().to(device)
            poor = inputs["poor_texture"].permute(0, 3, 1, 2).float().to(device)
            labels = labels.float().to(device)

            # Forward pass
            outputs = model(rich, poor)
            loss = criterion(outputs, labels.unsqueeze(1))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct_train += (predicted == labels.unsqueeze(1)).sum().item()
            total_train += labels.size(0)

            # Update progress bar
            train_loop.set_postfix(
                {
                    "loss": f"{train_loss / batch_idx:.4f}",
                    "acc": f"{correct_train / total_train:.4f}",
                }
            )

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        # Wrap validation loader with tqdm
        val_loop = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loop, start=1):
                inputs, labels = batch
                rich = inputs["rich_texture"].permute(0, 3, 1, 2).float().to(device)
                poor = inputs["poor_texture"].permute(0, 3, 1, 2).float().to(device)
                labels = labels.float().to(device)

                outputs = model(rich, poor)
                loss = criterion(outputs, labels.unsqueeze(1))

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct_val += (predicted == labels.unsqueeze(1)).sum().item()
                total_val += labels.size(0)

                # Update progress bar
                val_loop.set_postfix(
                    {
                        "loss": f"{val_loss / batch_idx:.4f}",
                        "acc": f"{correct_val / total_val:.4f}",
                    }
                )

        # Print epoch statistics
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(
            f"Train Loss: {train_loss / len(train_loader):.4f} | Acc: {correct_train / total_train:.4f}"
        )
        print(
            f"Val Loss: {val_loss / len(val_loader):.4f} | Acc: {correct_val / total_val:.4f}\n"
        )
