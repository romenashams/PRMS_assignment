from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch

from src.dataset import HippocampusDataset
from src.model import UNet3D


def overlay_mask(ax, base_img, mask, title):
    ax.imshow(base_img, cmap="gray")
    ax.imshow(mask, alpha=0.35, vmin=0, vmax=2)
    ax.set_title(title)
    ax.axis("off")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create results folder
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load validation dataset
    val_ds = HippocampusDataset(split_file="data/Task04_Hippocampus/splits/val.txt")

    # Pick a random validation sample
    idx = random.randint(0, len(val_ds) - 1)
    image, label = val_ds[idx]

    # Load model
    model = UNet3D(in_channels=1, num_classes=3).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    model.eval()

    # Inference
    with torch.no_grad():
        x = image.unsqueeze(0).to(device)  # (1, 1, D, H, W)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu()  # (D, H, W)

    # Choose middle slice in depth
    d = image.shape[1]
    mid = d // 2

    img_slice = image[0, mid].cpu().numpy()
    gt_slice = label[mid].cpu().numpy()
    pred_slice = pred[mid].cpu().numpy()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title("MRI (slice)")
    axes[0].axis("off")

    overlay_mask(axes[1], img_slice, gt_slice, "MRI + Ground Truth")
    overlay_mask(axes[2], img_slice, pred_slice, "MRI + Prediction")

    plt.tight_layout()

    # Save figure instead of showing
    out_path = results_dir / f"val_sample_{idx}_slice_{mid}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print("Saved visualization to:", out_path)


if __name__ == "__main__":
    main()
