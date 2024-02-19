import nibabel as nib
import numpy as np
from skimage import exposure, filters
from scipy import ndimage
import os

def load_mri_data(file_path):
    """Load MRI data using NiBabel."""
    img = nib.load(file_path)
    return img.get_fdata()

def rescale_intensity(volume):
    """Rescale intensity of MRI data to enhance contrast."""
    p2, p98 = np.percentile(volume, (2, 98))
    return exposure.rescale_intensity(volume, in_range=(p2, p98))

def gaussian_smoothing(volume, sigma=1):
    """Apply Gaussian smoothing to reduce noise."""
    return ndimage.gaussian_filter(volume, sigma=sigma)

def skull_stripping(volume):
    """Skull stripping to remove non-brain tissues."""
    # Here, we'll apply a simple thresholding method
    threshold = filters.threshold_otsu(volume)
    mask = volume > threshold
    return volume * mask

def save_mri_data(volume, output_path):
    """Save pre-processed MRI data."""
    img = nib.Nifti1Image(volume, np.eye(4))
    nib.save(img, output_path)

def preprocess_mri_scan(input_path, output_path, template_path=None):
    """Preprocess MRI scan."""
    # Load MRI data
    mri_data = load_mri_data(input_path)
    
    # Rescale intensity for better contrast
    mri_data = rescale_intensity(mri_data)
    
    # Apply Gaussian smoothing to reduce noise
    mri_data = gaussian_smoothing(mri_data)
    
    # Skull stripping to remove non-brain tissues
    mri_data = skull_stripping(mri_data)
    
    # Optional: Register MRI data to a template
    if template_path:
        mri_data = register_to_template(mri_data, template_path)
    
    # Save pre-processed MRI data
    save_mri_data(mri_data, output_path)

# Example usage:
input_file = "input_scan.nii.gz"
output_file = "preprocessed_scan.nii.gz"
template_file = "template.nii.gz"

preprocess_mri_scan(input_file, output_file, template_file)
