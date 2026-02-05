#Load images to verify shape, spacing, and label values
#before preprocessing to avoid silent bugs.

import nibabel as nib
import numpy as np

def load_nifti(path):
    nii = nib.load(path)
    data = nii.get_fdata()
    spacing = nii.header.get_zooms()
    return data, spacing
