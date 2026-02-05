from pathlib import Path
import random


DATA_DIR = Path("data/Task04_Hippocampus/")
IMAGES_DIR = DATA_DIR / "imagesTr"
LABELS_DIR = DATA_DIR / "labelsTr"
SPLITS_DIR = DATA_DIR / "splits"

VAL_RATIO = 0.2
SEED = 42


def main():
    if not IMAGES_DIR.exists():
        raise RuntimeError(f"Missing folder: {IMAGES_DIR}")
    if not LABELS_DIR.exists():
        raise RuntimeError(f"Missing folder: {LABELS_DIR}")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all training images
    image_files = sorted(IMAGES_DIR.glob("*.nii.gz"))
    if not image_files:
        raise RuntimeError(f"No .nii.gz files found in {IMAGES_DIR}")

    # Keep only filenames (not full paths)
    filenames = [p.name for p in image_files]


    # Shuffle deterministically
    random.seed(SEED)
    random.shuffle(filenames)

    # Split
    n_total = len(filenames)
    n_val = int(n_total * VAL_RATIO)
    val_files = sorted(filenames[:n_val])
    train_files = sorted(filenames[n_val:])

    # Save split lists
    train_path = SPLITS_DIR / "train.txt"
    val_path = SPLITS_DIR / "val.txt"

    train_path.write_text("\n".join(train_files) + "\n")
    val_path.write_text("\n".join(val_files) + "\n")

    print("✅ Split created successfully")
    print(f"Total: {n_total}")
    print(f"Train: {len(train_files)}")
    print(f"Val:   {len(val_files)}")
    print(f"\nSaved:")
    print(f" - {train_path}")
    print(f" - {val_path}")


if __name__ == "__main__":
    main()
