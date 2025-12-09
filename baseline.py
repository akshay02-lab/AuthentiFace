import os
import argparse
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


warnings.filterwarnings("ignore")


# =========================
#  Baseline CNN Model
# =========================

class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # input 224x224 -> 112 -> 56 -> 28
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 2)  # live / spoof

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 56, 56]
        x = self.pool(F.relu(self.conv3(x)))  # [B, 128, 28, 28]

        x = x.view(x.size(0), -1)       

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =========================
#  Evaluation
# =========================

def evaluate(model, loader, device, split_name="Val"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n=== {split_name} Evaluation ===")
    print(classification_report(all_labels, all_preds,
                                target_names=["live", "spoof"]))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


# =========================
#  Training
# =========================

def train(model, train_loader, val_loader, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")


        evaluate(model, val_loader, device, split_name="Val (epoch)")


# =========================
#  Main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Baseline CNN for LIVENESS_GROUP (train/val/test with live/spoof)."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to LIVENESS_GROUP directory (containing train/val/test)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )

    args = parser.parse_args()

    root_dir = args.root
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    test_dir = os.path.join(root_dir, "test")

    for d in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(d):
            raise ValueError(f"Expected folder not found: {d}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # simple normalization
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    # Datasets and loaders
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Val samples  : {len(val_dataset)}")
    print(f"[INFO] Test samples : {len(test_dataset)}")
    print(f"[INFO] Classes      : {train_dataset.classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Model
    model = BaselineCNN()
    print("[INFO] Starting training of baseline CNN...")
    train(model, train_loader, val_loader, device, epochs=args.epochs)

    print("\n[INFO] Final evaluation on TEST set:")
    evaluate(model, test_loader, device, split_name="Test")


if __name__ == "__main__":
    main()
