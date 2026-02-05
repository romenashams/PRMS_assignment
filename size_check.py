from pathlib import Path
import nibabel as nib
import numpy as np
from collections import Counter

DATA_DIR = Path("data/Task04_Hippocampus")
IMAGES_DIR = DATA_DIR / "imagesTr"

def get_shape_and_spacing(path: Path):
    nii = nib.load(str(path))
    shape = nii.shape  # (D, H, W) usually
    spacing = nii.header.get_zooms()  # (sx, sy, sz)
    return shape, spacing

def main():
    image_paths = sorted(IMAGES_DIR.glob("*.nii.gz"))

    if not image_paths:
        raise RuntimeError(f"No .nii.gz files found in: {IMAGES_DIR}")

    shapes = []
    spacings = []

    print(f"Found {len(image_paths)} images.\n")

    for p in image_paths:
        shape, spacing = get_shape_and_spacing(p)
        shapes.append(shape)
        spacings.append(spacing)

    shape_counter = Counter(shapes)
    spacing_counter = Counter(spacings)


    print(f"Unique shapes: {len(shape_counter)}")
    for shape, count in shape_counter.most_common(10):
        print(f"{shape}  -> {count} volumes")

    print(f"Unique spacings: {len(spacing_counter)}")
    for spacing, count in spacing_counter.most_common(10):
        print(f"{spacing}  -> {count} volumes")

    # Compute min/max dimensions
    shapes_arr = np.array(shapes)
    min_shape = shapes_arr.min(axis=0)
    max_shape = shapes_arr.max(axis=0)
    mean_shape = shapes_arr.mean(axis=0)


    print(f"Min shape: {tuple(min_shape)}")
    print(f"Max shape: {tuple(max_shape)}")
    print(f"Mean shape: {tuple(mean_shape.round(2))}")


if __name__ == "__main__":
    main()
