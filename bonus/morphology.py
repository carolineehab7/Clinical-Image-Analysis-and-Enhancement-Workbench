"""
Clinical Morphological Engine — Bonus Task.

All operations are implemented from scratch using only NumPy array slicing
and bitwise operations. No cv2, skimage, or scipy morphology functions are used.

Algorithm convention
--------------------
- Binary images are stored as uint8 with values 0 (background) or 255 (foreground).
- Erosion  : output pixel = 255 iff every foreground pixel of the SE is matched.
- Dilation : output pixel = 255 iff at least one foreground pixel of the SE is matched.
- Opening  : Erosion → Dilation   (removes small noise / thin protrusions).
- Closing  : Dilation → Erosion   (fills small holes / gaps).
- Boundary : Original − Eroded    (outline of anatomical structures).
"""

import numpy as np


# ─────────────────────────────────────────────
# Structuring-element factory
# ─────────────────────────────────────────────

def make_structuring_element(size: int, shape: str = "square") -> np.ndarray:
    """Return a binary structuring element (SE) of a given size and shape.

    Parameters
    ----------
    size : int
        Odd integer size of the SE (3, 5, 7, ...). Must be >= 1.
    shape : str
        "square" or "cross" (case-insensitive).

    Returns
    -------
    np.ndarray
        A 2D uint8 array with values in {0, 1}.

    Raises
    ------
    ValueError
        If size is not a positive odd integer or shape is unknown.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError("SE size must be a positive odd integer.")

    shape_norm = shape.lower().strip()
    se = np.zeros((size, size), dtype=np.uint8)
    if shape_norm == "square":
        se[:, :] = 1
    elif shape_norm == "cross":
        mid = size // 2
        se[mid, :] = 1   # horizontal bar
        se[:, mid] = 1   # vertical bar
    else:
        raise ValueError(f"Unknown SE shape: {shape!r}")
    return se


# ─────────────────────────────────────────────
# Thresholding
# ─────────────────────────────────────────────

def threshold(image: np.ndarray, T: int) -> np.ndarray:
    """Apply a global threshold to an image.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale or 3D RGB/RGBA image.
    T : int
        Threshold in [0, 255]. Values outside are clamped.

    Returns
    -------
    np.ndarray
        Binary uint8 image with values 0 or 255.

    Raises
    ------
    ValueError
        If the input image is empty or has unsupported shape.
    """
    gray = _to_gray(image)
    T = int(np.clip(T, 0, 255))
    return ((gray >= T).astype(np.uint8)) * 255


# ─────────────────────────────────────────────
# Core operations (vectorised "from scratch")
# ─────────────────────────────────────────────

def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert a 2D or 3D image to grayscale float64 for processing.

    Raises
    ------
    ValueError
        If the input is None, empty, not numeric, or has unsupported shape.
    """
    if image is None:
        raise ValueError("Image is None.")

    img = np.asarray(image)
    if img.size == 0:
        raise ValueError("Image is empty.")

    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        if img.shape[2] >= 3:
            gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        elif img.shape[2] == 1:
            gray = img[..., 0]
        else:
            raise ValueError("Unsupported channel count for image.")
    else:
        raise ValueError("Image must be 2D or 3D.")

    try:
        gray = gray.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Image must contain numeric values.") from exc

    return np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0)


def _to_binary(image: np.ndarray) -> np.ndarray:
    """Normalize an image to a strict {0,1} binary array."""
    gray = _to_gray(image)
    return (gray > 0).astype(np.uint8)


def _validate_se(se: np.ndarray) -> np.ndarray:
    """Validate and normalize a structuring element to a 2D uint8 mask."""
    if se is None:
        raise ValueError("Structuring element is None.")

    se_arr = np.asarray(se)
    if se_arr.size == 0:
        raise ValueError("Structuring element is empty.")
    if se_arr.ndim != 2:
        raise ValueError("Structuring element must be 2D.")
    if se_arr.shape[0] != se_arr.shape[1]:
        raise ValueError("Structuring element must be square.")
    if se_arr.shape[0] % 2 == 0:
        raise ValueError("Structuring element size must be odd.")

    return (se_arr > 0).astype(np.uint8)


def erode(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Perform binary erosion using the given structuring element.

    The output pixel is 1 only when all SE foreground positions match 1 in
    the padded input. This is implemented by AND-accumulating SE-shifted views.

    Parameters
    ----------
    binary : np.ndarray
        Input binary image (2D). If RGB, it is converted to grayscale first.
    se : np.ndarray
        Structuring element as a 2D array with values in {0, 1}.

    Returns
    -------
    np.ndarray
        Eroded binary image with values 0 or 255.
    """
    b = _to_binary(binary)
    se_bin = _validate_se(se)
    sh, sw = se_bin.shape
    ph, pw = sh // 2, sw // 2
    h, w = b.shape

    # Zero-pad (boundary = 0 → eroded pixels shrink inward)
    padded = np.pad(b, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)

    # Start with all-ones; AND in each shifted view where SE = 1
    result = np.ones((h, w), dtype=np.uint8)
    for i in range(sh):
        for j in range(sw):
            if se_bin[i, j]:
                result &= padded[i:i + h, j:j + w]

    return result * 255

def dilate(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Perform binary dilation using the given structuring element.

    The output pixel is 1 when any SE foreground position matches 1 in the
    padded input. This is implemented by OR-accumulating SE-shifted views.

    Parameters
    ----------
    binary : np.ndarray
        Input binary image (2D). If RGB, it is converted to grayscale first.
    se : np.ndarray
        Structuring element as a 2D array with values in {0, 1}.

    Returns
    -------
    np.ndarray
        Dilated binary image with values 0 or 255.
    """
    b = _to_binary(binary)
    se_bin = _validate_se(se)
    sh, sw = se_bin.shape
    ph, pw = sh // 2, sw // 2
    h, w = b.shape

    # Zero-pad (boundary = 0 → background stays background)
    padded = np.pad(b, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)

    # Start with all-zeros; OR in each shifted view where SE = 1
    result = np.zeros((h, w), dtype=np.uint8)
    for i in range(sh):
        for j in range(sw):
            if se_bin[i, j]:
                result |= padded[i:i + h, j:j + w]

    return result * 255


##### Compound Operations #####

# opening: Erosion then Dilation
# Removes small objects, breaks connections
def opening(binary: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Apply opening (erosion followed by dilation)."""
    return dilate(erode(binary, structuring_element), structuring_element)

# closing: Dilation then Erosion
# Fill small holes, connect broken parts
def closing(binary: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Apply closing (dilation followed by erosion)."""
    return erode(dilate(binary, structuring_element), structuring_element)

# boundary extraction: Original - Eroded
# Outlines structures
def boundary_extraction(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Extract boundaries by subtracting the eroded image from the original."""
    original_image = _to_binary(binary)
    eroded_image = _to_binary(erode(binary, se))
    return np.clip(original_image - eroded_image, 0, 1).astype(np.uint8) * 255
