"""
Image I/O Module
Handles loading DICOM, JPEG, BMP images and extracting metadata.
Saving processed images to disk.
"""

import os
import numpy as np
from PIL import Image

# Try importing pydicom for DICOM support
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False


def load_image(filepath: str):
    """
    Load an image from filepath.
    Returns: (image_array: np.ndarray uint8, metadata: dict)
    image_array is always uint8, grayscale or RGB.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext in ('.dcm', '.dicom') or ext == '':
        # Try DICOM first for extensionless files
        return _load_dicom(filepath)
    elif ext in ('.jpg', '.jpeg'):
        return _load_jpeg(filepath)
    elif ext == '.bmp':
        return _load_bmp(filepath)
    else:
        # Try to open with PIL as fallback
        try:
            return _load_pil(filepath, fmt=ext.lstrip('.').upper())
        except Exception:
            raise ValueError(f"Unsupported or unreadable format: {ext}")


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalize any numeric array to uint8 [0, 255]."""
    arr = array.astype(np.float64)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * 255.0
    else:
        arr = arr * 0  # All same value → black
    return arr.astype(np.uint8)


def _load_dicom(filepath: str):
    """Load a DICOM file. Returns (uint8 array, metadata dict)."""
    if not DICOM_AVAILABLE:
        raise ImportError("pydicom is required for DICOM files. Run: pip install pydicom")

    ds = pydicom.dcmread(filepath, force=True)

    # Extract pixel data
    try:
        pixel_array = ds.pixel_array
    except Exception as e:
        raise ValueError(f"Could not read pixel data from DICOM: {e}")
    
    if pixel_array.ndim == 3:
        # If last axis is channels (RGB/RGBA), keep the array
        if pixel_array.shape[2] in (3, 4):
            pass
        else:
            # Assume frames are the first axis: pick the middle frame
            mid = pixel_array.shape[0] // 2
            pixel_array = pixel_array[mid]

    # Handle different DICOM pixel representations
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    image_array = _normalize_to_uint8(pixel_array)

    # Build metadata dict safely
    def safe_get(tag, default='N/A'):
        try:
            val = getattr(ds, tag)
            return str(val) if val else default
        except AttributeError:
            return default

    h, w = image_array.shape[:2]
    metadata = {
        'Format':       'DICOM',
        'Width':        str(w),
        'Height':       str(h),
        'Bit Depth':    safe_get('BitsStored'),
        'Modality':     safe_get('Modality'),
        'Patient Name': safe_get('PatientName'),
        'Patient Age':  safe_get('PatientAge'),
        'Body Part':    safe_get('BodyPartExamined'),
        'Study Date':   safe_get('StudyDate'),
        'Institution':  safe_get('InstitutionName'),
    }
    return image_array, metadata


def _load_jpeg(filepath: str):
    """Load JPEG image."""
    img = Image.open(filepath)
    img.load()
    arr = np.array(img.convert('RGB') if img.mode not in ('L', 'RGB') else img, dtype=np.uint8)
    h, w = arr.shape[:2]
    metadata = {
        'Format':       'JPEG',
        'Width':        str(w),
        'Height':       str(h),
        'Bit Depth':    '8',
        'Modality':     'N/A',
        'Patient Name': 'N/A',
        'Patient Age':  'N/A',
        'Body Part':    'N/A',
        'Study Date':   'N/A',
        'Institution':  'N/A',
    }
    return arr, metadata


def _load_bmp(filepath: str):
    """Load BMP image."""
    img = Image.open(filepath)
    img.load()
    arr = np.array(img.convert('RGB') if img.mode not in ('L', 'RGB') else img, dtype=np.uint8)
    h, w = arr.shape[:2]
    metadata = {
        'Format':       'BMP',
        'Width':        str(w),
        'Height':       str(h),
        'Bit Depth':    '8',
        'Modality':     'N/A',
        'Patient Name': 'N/A',
        'Patient Age':  'N/A',
        'Body Part':    'N/A',
        'Study Date':   'N/A',
        'Institution':  'N/A',
    }
    return arr, metadata


def _load_pil(filepath: str, fmt: str):
    """Generic PIL loader as fallback."""
    img = Image.open(filepath)
    img.load()
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    metadata = {
        'Format':       fmt,
        'Width':        str(w),
        'Height':       str(h),
        'Bit Depth':    '8',
        'Modality':     'N/A',
        'Patient Name': 'N/A',
        'Patient Age':  'N/A',
        'Body Part':    'N/A',
        'Study Date':   'N/A',
        'Institution':  'N/A',
    }
    return arr, metadata


def save_image(image_array: np.ndarray, filepath: str) -> None:
    """
    Save a processed image to disk.
    Automatically infers format from extension. Supports PNG, JPEG, BMP, TIFF.
    """
    if image_array is None:
        raise ValueError("No image data to save.")

    # Ensure uint8
    if image_array.dtype != np.uint8:
        img_data = np.clip(image_array, 0, 255).astype(np.uint8)
    else:
        img_data = image_array.copy()

    # Convert numpy array to PIL Image
    if img_data.ndim == 2:
        pil_img = Image.fromarray(img_data, mode='L')
    elif img_data.ndim == 3 and img_data.shape[2] == 3:
        pil_img = Image.fromarray(img_data, mode='RGB')
    elif img_data.ndim == 3 and img_data.shape[2] == 4:
        pil_img = Image.fromarray(img_data, mode='RGBA')
    else:
        raise ValueError(f"Unsupported image shape for saving: {img_data.shape}")

    pil_img.save(filepath)
