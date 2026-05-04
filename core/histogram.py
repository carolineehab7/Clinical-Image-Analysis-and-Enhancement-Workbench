"""
Histogram Module — Implemented entirely from scratch.
Provides Local (Block-based) Histogram Equalization for contrast enhancement.
No built-in histogram equalization functions are used.
"""

import numpy as np


# ─────────────────────────────────────────────────────────
# Histogram Utilities
# ─────────────────────────────────────────────────────────

def compute_histogram(image: np.ndarray) -> np.ndarray:
    """
    Compute the intensity histogram of a grayscale image from scratch.

    We iterate over all 256 possible intensity levels and count pixels
    at each level. This is the explicit definition of a histogram.

    Args:
        image: Grayscale uint8 image (H, W).

    Returns:
        hist: 1D array of length 256, hist[i] = number of pixels with intensity i.
    """
    gray = _ensure_gray(image).flatten().astype(np.uint8)
    hist = np.zeros(256, dtype=np.int64)
    for intensity in range(256):
        hist[intensity] = np.sum(gray == intensity)
    return hist


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if RGB."""
    if image.ndim == 3:
        return (0.299 * image[:, :, 0] +
                0.587 * image[:, :, 1] +
                0.114 * image[:, :, 2]).astype(np.uint8)
    return image.astype(np.uint8)


# ─────────────────────────────────────────────────────────
# Block Equalization
# ─────────────────────────────────────────────────────────

def _equalize_block(block: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalisation to a single rectangular block.

    Algorithm:
        1. Build the local histogram H[i] for intensities 0..255.
        2. Compute the Cumulative Distribution Function (CDF):
               CDF[i] = sum of H[0..i]
        3. Find CDF_min (first non-zero CDF value).
        4. Apply the equalisation mapping:
               new_intensity = round( (CDF[i] - CDF_min) / (N - CDF_min) * 255 )
           where N = total number of pixels in the block.
        5. Map every pixel through this look-up table.

    Args:
        block: 2D uint8 sub-image.

    Returns:
        Equalised block with same shape as input, uint8.
    """
    flat = block.flatten().astype(np.uint8)
    n_pixels = flat.size

    # Step 1: Build histogram from scratch (explicit loop over intensity levels)
    hist = np.zeros(256, dtype=np.int64)
    for intensity in range(256):
        hist[intensity] = int(np.sum(flat == intensity))

    # Step 2: CDF (prefix sum of histogram)
    cdf = np.zeros(256, dtype=np.int64)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    # Step 3: Find the minimum non-zero CDF value
    cdf_min = 0
    for val in cdf:
        if val > 0:
            cdf_min = int(val)
            break

    denominator = n_pixels - cdf_min

    # If the block has no intensity spread, preserve it unchanged.
    # This avoids turning flat regions into black patches.
    if denominator <= 0:
        return block.astype(np.uint8).copy()

    # Step 4: Build look-up table (LUT)
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        val = int(round((cdf[i] - cdf_min) / denominator * 255))
        lut[i] = max(0, min(255, val))   # Clamp to valid uint8 range

    # Step 5: Apply LUT to every pixel in the block
    return lut[block.astype(np.uint8)]


# ─────────────────────────────────────────────────────────
# Local Histogram Equalization
# ─────────────────────────────────────────────────────────

def local_histogram_equalization(image: np.ndarray, block_size: int) -> np.ndarray:
    """
    Local (block-based) Histogram Equalization.

    The image is divided into non-overlapping blocks of size (block_size × block_size).
    Each block is equalised independently, so local contrast is enhanced even
    in regions with uneven illumination — critical for medical images.

    Args:
        image:      Input image (grayscale or RGB), uint8.
        block_size: Size of each local block in pixels (e.g., 8, 16, 32).

    Returns:
        Contrast-enhanced grayscale image, uint8, same spatial size as input.
    """
    gray = _ensure_gray(image)
    h, w = gray.shape
    result = np.zeros_like(gray, dtype=np.uint8)

    # Process each block
    for row_start in range(0, h, block_size):
        for col_start in range(0, w, block_size):
            row_end = min(row_start + block_size, h)
            col_end = min(col_start + block_size, w)

            block = gray[row_start:row_end, col_start:col_end]
            eq_block = _equalize_block(block)
            result[row_start:row_end, col_start:col_end] = eq_block

    return result
