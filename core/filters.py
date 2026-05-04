"""
Spatial Filters Module — All filter kernels and algorithms from scratch.

Includes:
- Average (box) filter
- Gaussian filter (kernel generated mathematically)
- Sobel edge detection (Gx, Gy, magnitude)
- Prewitt edge detection (Gx, Gy, magnitude)
- Median filter (sliding window, sort-based, from scratch)
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
from core.convolution import convolve2d, _convolve_single


# ─────────────────────────────────────────────────────────
# Kernel Generators
# ─────────────────────────────────────────────────────────

def make_average_kernel(size: int) -> np.ndarray:
    """
    Box / average filter kernel.
    Every entry = 1/(size*size), so the sum is always 1.
    """
    return np.ones((size, size), dtype=np.float64) / (size * size)


def make_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Gaussian filter kernel generated from the 2D Gaussian formula:
        G(x,y) = exp(-(x² + y²) / (2σ²))
    The kernel is then normalised so all weights sum to 1.

    Args:
        size:  Kernel dimension (must be odd).
        sigma: Standard deviation (controls blur strength).
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    half = size // 2
    # x and y offsets from the kernel centre
    x = np.arange(-half, half + 1, dtype=np.float64)
    y = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()   # Normalise
    return kernel


# ─────────────────────────────────────────────────────────
# Smoothing Filters
# ─────────────────────────────────────────────────────────

def average_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply an average (box) smoothing filter.

    Args:
        image:       Input uint8 image.
        kernel_size: Size of the square kernel (e.g., 3 for 3×3).

    Returns:
        Filtered image as uint8.
    """
    kernel = make_average_kernel(kernel_size)
    result = convolve2d(image.astype(np.float64), kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


def gaussian_filter(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """
    Apply a Gaussian smoothing filter.

    Args:
        image:       Input uint8 image.
        kernel_size: Size of the square kernel (must be odd).
        sigma:       Gaussian standard deviation.

    Returns:
        Filtered image as uint8.
    """
    kernel = make_gaussian_kernel(kernel_size, sigma)
    result = convolve2d(image.astype(np.float64), kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────
# Edge Detection
# ─────────────────────────────────────────────────────────

def _normalize_edge(arr: np.ndarray) -> np.ndarray:
    """Normalise a float edge map to uint8 [0, 255]."""
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale using luminosity weights."""
    if image.ndim == 3:
        # ITU-R BT.601 weights: 0.299 R + 0.587 G + 0.114 B
        return (0.299 * image[:, :, 0] +
                0.587 * image[:, :, 1] +
                0.114 * image[:, :, 2])
    return image.astype(np.float64)


def sobel_filter(image: np.ndarray):
    """
    Sobel edge detection.
    Uses 3×3 Sobel kernels:
        Kx = [[-1,0,1],[-2,0,2],[-1,0,1]]   (horizontal gradient)
        Ky = [[-1,-2,-1],[0,0,0],[1,2,1]]   (vertical gradient)

    Returns:
        (Gx, Gy, magnitude) — all as uint8 numpy arrays.
    """
    gray = _to_gray(image)

    # Sobel kernels (hand-crafted, not from any library)
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)

    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float64)

    Gx = _convolve_single(gray, Kx)
    Gy = _convolve_single(gray, Ky)
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

    return _normalize_edge(Gx), _normalize_edge(Gy), _normalize_edge(magnitude)


def prewitt_filter(image: np.ndarray):
    """
    Prewitt edge detection.
    Uses 3×3 Prewitt kernels:
        Kx = [[-1,0,1],[-1,0,1],[-1,0,1]]
        Ky = [[-1,-1,-1],[0,0,0],[1,1,1]]

    Returns:
        (Gx, Gy, magnitude) — all as uint8 numpy arrays.
    """
    gray = _to_gray(image)

    Kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float64)

    Ky = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]], dtype=np.float64)

    Gx = _convolve_single(gray, Kx)
    Gy = _convolve_single(gray, Ky)
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

    return _normalize_edge(Gx), _normalize_edge(Gy), _normalize_edge(magnitude)


# ─────────────────────────────────────────────────────────
# Median Filter (Non-linear, from scratch)
# ─────────────────────────────────────────────────────────

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
