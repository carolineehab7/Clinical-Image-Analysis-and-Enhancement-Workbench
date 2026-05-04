import numpy as np

def compute_histogram(image: np.ndarray) -> np.ndarray:
    gray = _ensure_gray(image).flatten().astype(np.uint8)
    hist = np.zeros(256, dtype=np.int64)
    for intensity in range(256):
        hist[intensity] = np.sum(gray == intensity)
    return hist


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return (0.299 * image[:, :, 0] +
                0.587 * image[:, :, 1] +
                0.114 * image[:, :, 2]).astype(np.uint8)
    return image.astype(np.uint8)

def _equalize_block(block: np.ndarray) -> np.ndarray:
    flat = block.flatten().astype(np.uint8)
    n_pixels = flat.size

    hist = np.zeros(256, dtype=np.int64)
    for intensity in range(256):
        hist[intensity] = int(np.sum(flat == intensity))

    cdf = np.zeros(256, dtype=np.int64)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    cdf_min = 0
    for val in cdf:
        if val > 0:
            cdf_min = int(val)
            break

    denominator = n_pixels - cdf_min

    if denominator <= 0:
        return block.astype(np.uint8).copy()

    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        val = int(round((cdf[i] - cdf_min) / denominator * 255))
        lut[i] = max(0, min(255, val))

    return lut[block.astype(np.uint8)]

def local_histogram_equalization(image: np.ndarray, block_size: int) -> np.ndarray:
    gray = _ensure_gray(image)
    h, w = gray.shape
    result = np.zeros_like(gray, dtype=np.uint8)

    for row_start in range(0, h, block_size):
        for col_start in range(0, w, block_size):
            row_end = min(row_start + block_size, h)
            col_end = min(col_start + block_size, w)

            block = gray[row_start:row_end, col_start:col_end]
            eq_block = _equalize_block(block)
            result[row_start:row_end, col_start:col_end] = eq_block

    return result
