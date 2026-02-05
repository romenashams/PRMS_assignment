# src/preprocessing.py
import numpy as np

# Z-score normalization of non-zero voxels
#To avoids background zeros dominating mean/std.
def zscore(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:

    v = volume.astype(np.float32, copy=False)
    mask = v != 0

    # If everything is zero (rare), just return as-is
    if not np.any(mask):
        return v

    mean = v[mask].mean()
    std = v[mask].std()

    # Avoid division by zero
    if std < eps:
        return v - mean

    v = (v - mean) / (std + eps)
    return v

# Center crop or symmetric pad to target shape.
#we chose target shape(43, 59, 47), as MAX size, so
#this will mostly pad (not crop), but cropping is here for safety.
def crop_center(volume: np.ndarray, target_shape: tuple[int, int, int], pad_value: float = 0) -> np.ndarray:

    assert volume.ndim == 3, f"Expected 3D volume, got shape {volume.shape}"
    out = volume

    # --- Center crop if needed ---
    for axis in range(3):
        if out.shape[axis] > target_shape[axis]:
            start = (out.shape[axis] - target_shape[axis]) // 2
            end = start + target_shape[axis]
            slicer = [slice(None)] * 3
            slicer[axis] = slice(start, end)
            out = out[tuple(slicer)]

    # --- Symmetric pad if needed ---
    pad_width = []
    for axis in range(3):
        diff = target_shape[axis] - out.shape[axis]
        if diff > 0:
            before = diff // 2
            after = diff - before
            pad_width.append((before, after))
        else:
            pad_width.append((0, 0))

    out = np.pad(out, pad_width, mode="constant", constant_values=pad_value)

    if out.shape != target_shape:
        raise ValueError(f"crop_center failed: got {out.shape}, expected {target_shape}")

    return out

#Full preprocessing pipeline
def preprocess_pair(
    image: np.ndarray,
    label: np.ndarray,
    target_shape: tuple[int, int, int] = (43, 59, 47),
) -> tuple[np.ndarray, np.ndarray]:
    
    if image.shape != label.shape:
        raise ValueError(f"Image/label shape mismatch: {image.shape} vs {label.shape}")

    image_p = zscore(image)
    image_p = crop_center(image_p, target_shape=target_shape, pad_value=0)

    # Keep labels as integer class IDs (0/1/2), padding with 0 (background)
    label_p = crop_center(label, target_shape=target_shape, pad_value=0)

    return image_p, label_p



