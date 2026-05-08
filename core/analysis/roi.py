import numpy as np


def extract_roi(image, x0, y0, x1, y1):
    row0, row1 = min(y0, y1), max(y0, y1)
    col0, col1 = min(x0, x1), max(x0, x1)
    h, w = image.shape[:2]
    row0, col0 = max(0, row0), max(0, col0)
    row1, col1 = min(h, row1), min(w, col1)

    if row1 <= row0 or col1 <= col0:
        return None

    patch = image[row0:row1, col0:col1]
    if patch.ndim == 3:
        patch = (0.299 * patch[..., 0] + 0.587 * patch[..., 1] + 0.114 * patch[..., 2])
    return np.clip(patch, 0, 255).astype(np.uint8)
