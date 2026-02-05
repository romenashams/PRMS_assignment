from pathlib import Path
import nibabel as nib
import torch
from torch.utils.data import Dataset

from src.preprocessing import preprocess_pair

#Loads a nii file and returns a numpy array (D,H,W).
def load_nifti(path: Path):
    nii = nib.load(str(path))
    return nii.get_fdata()

#Loads one MRI volume and its mask, applies preprocessing, and returns PyTorch tensors for training.
class HippocampusDataset(Dataset):
    def __init__(self, data_dir="data/Task04_Hippocampus/", split_file="data/Task04_Hippocampus/splits/train.txt", target_shape=(43, 59, 47)):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "imagesTr"
        self.labels_dir = self.data_dir / "labelsTr"

        self.target_shape = target_shape

        # Read filenames from split file
        split_path = Path(split_file)
        if not split_path.exists():
            raise RuntimeError(f"Split file not found: {split_path}")

        self.filenames = split_path.read_text().splitlines()

        if len(self.filenames) == 0:
            raise RuntimeError(f"Split file is empty: {split_path}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        image_path = self.images_dir / fname
        label_path = self.labels_dir / fname

        image = load_nifti(image_path)
        label = load_nifti(label_path)

        image, label = preprocess_pair(image, label, target_shape=self.target_shape)

        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        # Label should be class ids (0,1,2)
        label = torch.tensor(label, dtype=torch.long)

        return image, label



# from torch.utils.data import DataLoader
# from src.dataset import HippocampusDataset

# ds = HippocampusDataset(split_file="data/Task04_Hippocampus/splits/train.txt")
# dl = DataLoader(ds, batch_size=2, shuffle=True)

# images, labels = next(iter(dl))

# print("Batch images:", images.shape)
# print("Batch labels:", labels.shape)
# print("Label unique values:", labels.unique())