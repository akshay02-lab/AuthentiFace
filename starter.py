"""
starter.py


RAF_DB/
  DATASET/
    train/1..7
    test/1..7

SAMM_v1/
  Anger/
  Contempt/
  Disgust/
  Fear/
  Happiness/
  Other/
  Sadness/
  Surprise/

LIVENESS/
  train/live, spoof
  val/live, spoof
  test/live, spoof

Run:
    conda activate dl_env_clean
    cd "<PROJECT_ROOT>"
    python starter.py
    LIVENESS_CLASSES
"""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import timm


# ==========================
#  GLOBAL CONFIG
# ==========================

PROJECT_ROOT = Path(".")

RAF_TRAIN_DIR = PROJECT_ROOT / "RAF_GROUP" / "DATASET" / "train"
RAF_TEST_DIR = PROJECT_ROOT / "RAF_GROUP" / "DATASET" / "test"

SAMM_DIR = PROJECT_ROOT / "SAMM_v1"

LIVENESS_ROOT = PROJECT_ROOT / "LIVENESS_GROUP"

WEIGHTS_DIR = PROJECT_ROOT / "WEIGHTS_GROUP"
WEIGHTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

EPOCHS_RAF = 15
EPOCHS_SAMM = 10
EPOCHS_LIVE = 8

LR_RAF = 3e-5
LR_SAMM = 1e-5
LR_LIVE = 1e-4

SEED = 42


# ==========================
#  UTILITIES
# ==========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tf, val_tf


# ==========================
#  MODEL: DUAL-HEAD VIT
# ==========================

class DualHeadViT(nn.Module):
    """
    ViT backbone + two heads:
      - emotion_head: num_emotions classes (macro/micro expressions)
      - liveness_head: num_liveness classes (real/spoof)
    """

    def __init__(
        self,
        backbone_name: str = "vit_tiny_patch16_224",
        num_emotions: int = 8,
        num_liveness: int = 2,
        pretrained_backbone: bool = True,
    ):
        super().__init__()

   
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,
            num_classes=0,
            global_pool="avg",
        )
        embed_dim = self.backbone.num_features

        self.emotion_head = nn.Linear(embed_dim, num_emotions)
        self.liveness_head = nn.Linear(embed_dim, num_liveness)

    def forward(self, x):
        feats = self.backbone(x)  # [B, D]
        emo_logits = self.emotion_head(feats)      
        live_logits = self.liveness_head(feats)  
        return emo_logits, live_logits


# ==========================
#  DATA LOADERS
# ==========================

def build_raf_loaders(
    batch_size: int,
    num_workers: int,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    RAF_DB is already split into train/test folders: 1..7.
    We split RAF train into train/val (e.g., 85/15).
    """
    train_tf, val_tf = get_transforms()

    full_train = datasets.ImageFolder(str(RAF_TRAIN_DIR), transform=train_tf)
    test_ds = datasets.ImageFolder(str(RAF_TEST_DIR), transform=val_tf)

    classes = full_train.classes  # ['1','2',...,'7']

    val_ratio = 0.15
    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    set_seed(seed)
    train_ds, val_ds = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    print("[RAF_DB] Train/Val/Test sizes:",
          len(train_ds), len(val_ds), len(test_ds))
    print("[RAF_DB] Classes (folder labels):", classes)

    return train_loader, val_loader, test_loader, classes


def build_samm_loaders(
    batch_size: int,
    num_workers: int,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    SAMM_v1 is a folder with subfolders per class.
    We create train/val/test splits as 70/15/15.
    """
    train_tf, val_tf = get_transforms()

    samm_full_train = datasets.ImageFolder(str(SAMM_DIR), transform=train_tf)
    samm_full_eval = datasets.ImageFolder(str(SAMM_DIR), transform=val_tf)

    classes = samm_full_train.classes
    assert classes == samm_full_eval.classes

    n = len(samm_full_train)
    val_ratio = 0.15
    test_ratio = 0.15
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    indices = list(range(n))
    set_seed(seed)
    random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    samm_train_ds = Subset(samm_full_train, train_idx)
    samm_val_ds = Subset(samm_full_eval, val_idx)
    samm_test_ds = Subset(samm_full_eval, test_idx)

    train_loader = DataLoader(samm_train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(samm_val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(samm_test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    print("[SAMM] Train/Val/Test sizes:",
          len(samm_train_ds), len(samm_val_ds), len(samm_test_ds))
    print("[SAMM] Classes:", classes)

    return train_loader, val_loader, test_loader, classes


def build_liveness_loaders(
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Liveness dataset is expected as:
        LIVENESS/train/live,spoof
        LIVENESS/val/live,spoof
        LIVENESS/test/live,spoof
    """
    train_tf, val_tf = get_transforms()

    train_dir = LIVENESS_ROOT / "train"
    val_dir = LIVENESS_ROOT / "val"
    test_dir = LIVENESS_ROOT / "test"

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)
    test_ds = datasets.ImageFolder(str(test_dir), transform=val_tf)

    assert train_ds.classes == val_ds.classes == test_ds.classes
    classes = train_ds.classes  # ['live', 'spoof']

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    print("[LIVENESS] Train/Val/Test sizes:",
          len(train_ds), len(val_ds), len(test_ds))
    print("[LIVENESS] Classes:", classes)

    return train_loader, val_loader, test_loader, classes


# ==========================
#  TRAINING / EVAL HELPERS
# ==========================

def train_emotion_only(
    model: DualHeadViT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    save_path: Path,
) -> None:
    """
    Train backbone + emotion_head only (ignore liveness head).
    """
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # ------ Train ------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            emo_logits, _ = model(x)
            loss = criterion(emo_logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = emo_logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ------ Val ------
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                emo_logits, _ = model(x)
                loss = criterion(emo_logits, y)
                val_loss_sum += loss.item() * x.size(0)
                preds = emo_logits.argmax(1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        print(f"[EMO] Epoch {epoch+1}/{num_epochs} "
              f"Train {train_loss:.4f}/{train_acc:.4f} | "
              f"Val {val_loss:.4f}/{val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("   → Saved best emotion model to", save_path)


def eval_emotion(
    model: DualHeadViT,
    test_loader: DataLoader,
    class_names: List[str],
) -> None:
    model.to(DEVICE)
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            emo_logits, _ = model(x)
            preds = emo_logits.argmax(1)
            all_true.extend(y.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    print("=== Emotion head results ===")
    print(classification_report(all_true, all_pred, target_names=class_names))
    cm = confusion_matrix(all_true, all_pred)
    print("Confusion matrix:\n", cm)


def train_liveness_head(
    model: DualHeadViT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    save_path: Path,
    freeze_backbone: bool = True,
    freeze_emotion_head: bool = True,
) -> None:
    """
    Train only liveness_head (optionally unfreezing some of the backbone).
    """
    model.to(DEVICE)


    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    if freeze_emotion_head:
        for p in model.emotion_head.parameters():
            p.requires_grad = False

    live_params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(live_params, lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # ------ Train ------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            _, live_logits = model(x)
            loss = criterion(live_logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = live_logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ------ Val ------
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                _, live_logits = model(x)
                loss = criterion(live_logits, y)
                val_loss_sum += loss.item() * x.size(0)
                preds = live_logits.argmax(1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        print(f"[LIVE] Epoch {epoch+1}/{num_epochs} "
              f"Train {train_loss:.4f}/{train_acc:.4f} | "
              f"Val {val_loss:.4f}/{val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("   → Saved best dual-head (with liveness) model to", save_path)


def eval_liveness(
    model: DualHeadViT,
    test_loader: DataLoader,
    class_names: List[str],
) -> None:
    model.to(DEVICE)
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, live_logits = model(x)
            preds = live_logits.argmax(1)
            all_true.extend(y.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    print("=== Liveness head results ===")
    print(classification_report(all_true, all_pred, target_names=class_names))
    cm = confusion_matrix(all_true, all_pred)
    print("Confusion matrix:\n", cm)


# ==========================
#  MAIN PIPELINE
# ==========================

def main():
    set_seed(SEED)
    print("Using device:", DEVICE)

    # ----- STEP 1: RAF_DB pretraining (emotion only) -----
    print("\nSTEP 1: Pretrain ViT backbone on RAF_DB (emotion head)")
    raf_train_loader, raf_val_loader, raf_test_loader, raf_classes = build_raf_loaders(
        BATCH_SIZE, NUM_WORKERS, seed=SEED
    )

    raf_model = DualHeadViT(
        backbone_name="vit_tiny_patch16_224",
        num_emotions=len(raf_classes),  # 7
        num_liveness=2,
        pretrained_backbone=True,
    )

    raf_emo_ckpt = WEIGHTS_DIR / "dual_head_raf_emo.pth"
    train_emotion_only(
        raf_model,
        raf_train_loader,
        raf_val_loader,
        num_epochs=EPOCHS_RAF,
        lr=LR_RAF,
        save_path=raf_emo_ckpt,
    )

    raf_model.load_state_dict(torch.load(raf_emo_ckpt, map_location=DEVICE))
    eval_emotion(raf_model, raf_test_loader, class_names=raf_classes)

    print("\nSTEP 2: Skipping SAMM fine-tuning; using RAF_DB model for emotions.")
    samm_classes = raf_classes         
    samm_emo_ckpt = raf_emo_ckpt      


    # ----- STEP 3: Train liveness head on LIVENESS dataset -----
    print("\nSTEP 3: Train liveness head on LIVENESS (CelebA-Spoof derived)")
    if not LIVENESS_ROOT.exists():
        print(
            f"[WARNING] LIVENESS folder not found at {LIVENESS_ROOT}. "
            "Run liveness.py first to create it."
        )
        return

    live_train_loader, live_val_loader, live_test_loader, live_classes = build_liveness_loaders(
        BATCH_SIZE, NUM_WORKERS
    )

    live_model = DualHeadViT(
        backbone_name="vit_tiny_patch16_224",
        num_emotions=len(samm_classes),
        num_liveness=len(live_classes),
        pretrained_backbone=False,
    )

    samm_state = torch.load(samm_emo_ckpt, map_location=DEVICE)
    live_model.load_state_dict(samm_state, strict=False)

    live_ckpt = WEIGHTS_DIR / "dual_head_samm_liveness.pth"
    train_liveness_head(
        live_model,
        live_train_loader,
        live_val_loader,
        num_epochs=EPOCHS_LIVE,
        lr=LR_LIVE,
        save_path=live_ckpt,
        freeze_backbone=True,
        freeze_emotion_head=True,
    )

    live_model.load_state_dict(torch.load(live_ckpt, map_location=DEVICE))
    eval_liveness(live_model, live_test_loader, class_names=live_classes)

    print("\nPipeline complete.")
    print("Emotion + micro-expression model:", samm_emo_ckpt)
    print("Dual-head (emotion + liveness) model:", live_ckpt)


if __name__ == "__main__":
    main()

