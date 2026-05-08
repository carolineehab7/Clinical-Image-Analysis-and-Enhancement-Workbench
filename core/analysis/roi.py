import numpy as np


def extract_roi(image, x0, y0, x1, y1):
    """Extract a rectangular ROI from image, returned as a grayscale uint8 array.

    Coordinates are in image-pixel space. Returns None if the region is empty.
    """
    r0, r1 = min(y0, y1), max(y0, y1)
    c0, c1 = min(x0, x1), max(x0, x1)
    h, w = image.shape[:2]
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(h, r1), min(w, c1)

    if r1 <= r0 or c1 <= c0:
        return None

    patch = image[r0:r1, c0:c1]
    if patch.ndim == 3:
        patch = (0.299 * patch[..., 0] + 0.587 * patch[..., 1] + 0.114 * patch[..., 2])
    return np.clip(patch, 0, 255).astype(np.uint8)
