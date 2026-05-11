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
    """Return a binary SE of given size and shape.

    Parameters
    ----------
    size  : odd integer (3, 5, 7, …)
    shape : 'square' or 'cross'
    """
    se = np.zeros((size, size), dtype=np.uint8)
    if shape == "square":
        se[:, :] = 1
    elif shape == "cross":
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
    """Global threshold: pixels ≥ T → 255, else 0. Converts to grayscale first."""
    if image.ndim == 3:
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    else:
        gray = image.astype(np.float64)
    return ((gray >= T).astype(np.uint8)) * 255


# ─────────────────────────────────────────────
# Core operations (vectorised "from scratch")
# ─────────────────────────────────────────────

def _to_binary(image: np.ndarray) -> np.ndarray:
    """Normalise any uint8 image to a strict {0,1} binary array."""
    return (image > 0).astype(np.uint8)


def erode(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Binary erosion.

    For each output pixel (r, c): set to 1 iff ALL pixels under the SE in the
    padded input are 1.  Implemented by AND-accumulating SE-shifted views.
    """
    b = _to_binary(binary)
    sh, sw = se.shape
    ph, pw = sh // 2, sw // 2
    h, w   = b.shape

    # Zero-pad (boundary = 0 → eroded pixels shrink inward)
    padded = np.pad(b, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)

    # Start with all-ones; AND in each shifted view where SE = 1
    result = np.ones((h, w), dtype=np.uint8)
    for i in range(sh):
        for j in range(sw):
            if se[i, j]:
                result &= padded[i:i + h, j:j + w]

    return result * 255


def dilate(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Binary dilation.

    For each output pixel (r, c): set to 1 iff ANY pixel under the SE in the
    padded input is 1.  Implemented by OR-accumulating SE-shifted views.
    """
    b = _to_binary(binary)
    sh, sw = se.shape
    ph, pw = sh // 2, sw // 2
    h, w   = b.shape

    # Zero-pad (boundary = 0 → background stays background)
    padded = np.pad(b, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)

    # Start with all-zeros; OR in each shifted view where SE = 1
    result = np.zeros((h, w), dtype=np.uint8)
    for i in range(sh):
        for j in range(sw):
            if se[i, j]:
                result |= padded[i:i + h, j:j + w]

    return result * 255


##### Compound Operations #####

# opening: Erosion then Dilation
# Removes small objects, breaks connections
def opening(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    return dilate(erode(binary, se), se)

# closing: Dilation then Erosion
# Fill small holes, connect broken parts
def closing(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    return erode(dilate(binary, se), se)

# boundary extraction: Original - Eroded
# Outlines structures
def boundary_extraction(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    original_image = _to_binary(binary)
    eroded_image = _to_binary(erode(binary, se))
    return np.clip(original_image - eroded_image, 0, 1).astype(np.uint8) * 255
