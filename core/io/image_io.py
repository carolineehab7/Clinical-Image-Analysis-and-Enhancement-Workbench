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

    ext = os.path.splitext(filepath)[1].lower()    #splits path into file name + extension

    if ext in ('.dcm', '.dicom') or ext == '':       # Try DICOM first for extensionless files
        return _load_dicom(filepath)
    elif ext in ('.jpg', '.jpeg'):
        return _load_jpeg(filepath)
    elif ext == '.bmp':
        return _load_bmp(filepath)
    else:                  # if not dicom,jpeg,bmp Try to open with generic PIL loader as fallback
        try:
            return _load_pil(filepath, fmt=ext.lstrip('.').upper())
        except Exception:
            raise ValueError(f"Unsupported or unreadable format: {ext}")



def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalize any numeric array to uint8 [0, 255]."""
    arr = array.astype(np.float64)   #coverts image array to floats
    mn, mx = arr.min(), arr.max()     #finds min and max pixel values in the img
    if mx > mn:     #checks that the img is not constant
        arr = (arr - mn) / (mx - mn) * 255.0   #normalization formula
    else:
        arr = arr * 0  # All same value → black to avoid division by zero
    return arr.astype(np.uint8)   #stores values from 0 to 255 suitable for img display



def _load_dicom(filepath: str):
    """Load a DICOM file. Returns (uint8 array, metadata dict)."""
    if not DICOM_AVAILABLE: #checks if pydicom was imported successfully
        raise ImportError("pydicom is required for DICOM files. Run: pip install pydicom")

    ds = pydicom.dcmread(filepath, force=True)   #reads the dicom file

    # Extracts img pixels from dicom dataset
    try:
        pixel_array = ds.pixel_array
    except Exception as e:
        raise ValueError(f"Could not read pixel data from DICOM: {e}")

    # Handle different DICOM pixel representations
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):  #to convert raw stored pixel values into more meaningful medical intensity values
        pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    image_array = _normalize_to_uint8(pixel_array)



    # Build metadata dict safely


    def safe_get(tag, default='N/A'):  #to safely extract metadata from the dicom set cause not all dicom files contain all tags
        try:
            val = getattr(ds, tag)
            return str(val) if val else default  #returns tag as string and N/A if missing to avoid crashing
        except AttributeError:
            return default

    h, w = image_array.shape[:2]   #gets height,width of the img
    metadata = {             #creates dictionary containing dicom metadata
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
    return image_array, metadata # returns dicom img after normalization and the metadata to the GUI


def _bits_for_mode(mode: str) -> str:
    return {"1": "1", "L": "8", "P": "8", "RGB": "24", "RGBA": "32",
            "I": "32", "F": "32"}.get(mode, "?")


def _load_pil(filepath: str, fmt: str = "Unknown"):
    """Generic PIL-based loader. Returns (uint8 array, metadata dict)."""
    img = Image.open(filepath).convert("RGB") if Image.open(filepath).mode not in ("L", "RGB") else Image.open(filepath)
    arr = _normalize_to_uint8(np.array(img))
    h, w = arr.shape[:2]
    metadata = {
        "Format":       fmt,
        "Width":        str(w),
        "Height":       str(h),
        "Bit Depth":    f"{_bits_for_mode(img.mode)} bit",
        "Modality":     "N/A",
        "Patient Name": "N/A",
        "Patient Age":  "N/A",
        "Body Part":    "N/A",
        "Study Date":   "N/A",
        "Institution":  "N/A",
    }
    return arr, metadata


def _load_jpeg(filepath: str):
    """Load a JPEG image. Returns (uint8 array, metadata dict)."""
    return _load_pil(filepath, fmt="JPEG")


def _load_bmp(filepath: str):
    """Load a BMP image. Returns (uint8 array, metadata dict)."""
    return _load_pil(filepath, fmt="BMP")


def save_image(image: np.ndarray, filepath: str) -> None:
    """
    Save an image to disk.
    
    Arguments:
        image: Input array (uint8) that will be saved.
        filepath: Output file path.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a NumPy array.")
    
    # Ensure the image is uint8
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image and save
    if image.ndim == 2:
        # Grayscale
        pil_image = Image.fromarray(image, mode='L')
    elif image.ndim == 3 and image.shape[2] == 3:
        # RGB
        pil_image = Image.fromarray(image, mode='RGB')
    elif image.ndim == 3 and image.shape[2] == 4:
        # RGBA
        pil_image = Image.fromarray(image, mode='RGBA')
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    pil_image.save(filepath)

