import numpy as np
from numpy.lib.stride_tricks import as_strided

def median_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Median filter — non-linear, ideal for salt-and-pepper noise.

    Implementation: stride-trick sliding window extracts all overlapping
    (kernel_size × kernel_size) patches, sorts each patch, and picks the
    middle value. No scipy or cv2 functions are used.

    Args:
        image:       Input uint8 image.
        kernel_size: Size of the square neighbourhood (e.g., 3 for 3×3).

    Returns:
        Filtered image as uint8.
    """
    if image.ndim == 3:
        channels = [_median_single(image[:, :, c], kernel_size)
                    for c in range(image.shape[2])]
        return np.stack(channels, axis=2)
    return _median_single(image, kernel_size)


def _median_single(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply median filter to a single 2D channel."""
    h, w = image.shape
    pad = kernel_size // 2

    # Reflect-pad to handle borders
    padded = np.pad(image.astype(np.float64), pad, mode='reflect')

    # Build (H, W, kSize, kSize) patch view using stride tricks
    shape   = (h, w, kernel_size, kernel_size)
    s       = padded.strides
    strides = (s[0], s[1], s[0], s[1])
    patches = as_strided(padded, shape=shape, strides=strides)

    # Flatten each patch to 1D, sort it, pick the middle element
    n = kernel_size * kernel_size
    mid_idx = n // 2
    flat_patches = patches.reshape(h, w, n)

    # Sort each patch independently — this IS the median computation
    sorted_patches = np.sort(flat_patches, axis=2)
    result = sorted_patches[:, :, mid_idx]

    return np.clip(result, 0, 255).astype(np.uint8)
