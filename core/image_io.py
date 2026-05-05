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



def _load_jpeg(filepath: str):
    """Load JPEG image."""
    img = Image.open(filepath)  #opens jpeg file using pillow
    img.load()    #forces img data to be loaded into memory
    arr = np.array(img.convert('RGB') if img.mode not in ('L', 'RGB') else img, dtype=np.uint8)   #coverts img to numpy array
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
    return arr, metadata  #returns jpeg img as numpy array and metadats dictionary



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
    if image_array is None:  #checks if there is img
        raise ValueError("No image data to save.")

    # Ensure uint8
    if image_array.dtype != np.uint8:  #checks img is from 0 to 255
        img_data = np.clip(image_array, 0, 255).astype(np.uint8)
    else:
        img_data = image_array.copy()     #if img is already 0 to 255 it just makes a copy 

    # Convert numpy array to PIL Image
    if img_data.ndim == 2:          #checks if img is grayscale
        pil_img = Image.fromarray(img_data, mode='L')  # converts grayscale numpy array into a PIL image
    elif img_data.ndim == 3 and img_data.shape[2] == 3:  #checks if img is RGB
        pil_img = Image.fromarray(img_data, mode='RGB')
    elif img_data.ndim == 3 and img_data.shape[2] == 4:  #checks if img is RGBA
        pil_img = Image.fromarray(img_data, mode='RGBA')
    else:
        raise ValueError(f"Unsupported image shape for saving: {img_data.shape}")

    pil_img.save(filepath)
