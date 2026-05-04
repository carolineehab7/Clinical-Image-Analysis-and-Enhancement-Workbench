import numpy as np
from core.convolution import convolve2d

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
