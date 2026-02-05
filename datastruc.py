from pathlib import Path
from src.dataset_util import load_nifti
from src.preprocessing import preprocess_pair

# Load a sample image and label to verify preprocessing

image_path = Path("data/Task04_Hippocampus/imagesTr/hippocampus_123.nii.gz")
label_path = Path("data/Task04_Hippocampus/labelsTr/hippocampus_123.nii.gz")

image, img_spacing = load_nifti(image_path)
label, lbl_spacing = load_nifti(label_path)

print("Image shape:", image.shape)
print("Label shape:", label.shape)
print("Image spacing:", img_spacing)
print("Label spacing:", lbl_spacing)
print("Unique labels:", set(label.flatten()))


image_p, label_p = preprocess_pair(image, label)

print("Original image shape:", image.shape)
print("Original label shape:", label.shape)

print("Processed image shape:", image_p.shape)
print("Processed label shape:", label_p.shape)

print("Label values:", set(label_p.flatten()))

