from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import HippocampusDataset
from src.model import UNet3D

 
# pred: (B, D, H, W) predicted class ids
# target: (B, D, H, W) ground truth class ids
# Computes mean Dice over foreground classes (1..num_classes-1).

def dice_score(pred, target, num_classes=3, eps=1e-6):

    dices = []

    for cls in range(1, num_classes):  # skip background=0
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        dice = (2.0 * intersection + eps) / (union + eps)
        dices.append(dice.item())

    return sum(dices) / len(dices)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)  # (B,1,D,H,W)
        labels = labels.to(device)  # (B,D,H,W)

        logits = model(images)      # (B,C,D,H,W)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device, num_classes=3):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)  # (B,D,H,W)

        total_loss += loss.item()
        total_dice += dice_score(preds, labels, num_classes=num_classes)

    return total_loss / len(loader), total_dice / len(loader)


def train(
    data_dir="data/Task04_Hippocampus/",
    train_split="data/Task04_Hippocampus/splits/train.txt",
    val_split="data/Task04_Hippocampus/splits/val.txt",
    num_classes=3,
    batch_size=2,
    lr=1e-3,
    epochs=15,
    save_path="checkpoints/best_model.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = HippocampusDataset(data_dir=data_dir, split_file=train_split)
    val_ds = HippocampusDataset(data_dir=data_dir, split_file=val_split)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = UNet3D(in_channels=1, num_classes=num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    best_val_dice = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice = validate(model, val_loader, loss_fn, device, num_classes=num_classes)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Saved new best model (Dice={best_val_dice:.4f})")

    print("\nTraining finished.")
    print("Best validation Dice:", best_val_dice)
    print("Model saved at:", save_path)
