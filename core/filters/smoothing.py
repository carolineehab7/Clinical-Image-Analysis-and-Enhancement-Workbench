"Building smoothing Linear filters from scratch"
import numpy as np
from .convolution import MainConv

def avg_kernel(size: int) -> np.ndarray:
    """
    Box average filter kernel.
    Every entry = 1/(size*size), so the sum is always 1.
    """
    return np.ones((size, size), dtype=np.float64) / (size * size)


def gauss_kernel(size: int, variance: float) -> np.ndarray:
    """
    Gaussian filter kernel generated from the 2D Gaussian formula:
        G(x,y) = exp(-(x² + y²) / (2σ²))
    where σ² is the user-supplied variance.
    The kernel is then normalised so all weights sum to 1.

    Arguments:
        size:  Kernel dimension selected by user.
        variance: Gaussian variance it controls the spread / smoothness.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    if variance <= 0:
        raise ValueError("Variance must be positive.")

    half = size // 2
    # Create a grid of (x, y) coordinates centred at zero for the kernel size.
    x = np.arange(-half, half + 1, dtype=np.float64)
    y = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * variance))
    kernel /= kernel.sum() #Normalization divides the whole kernel by its sum so that the image doesn't get brighter or darker after filtering.
    return kernel

def box_smoothing_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply an average (box) smoothing filter.
    Arguments:
        image:       Input uint8 image.
        kernel_size: Size of the square kernel (e.g., 3 for 3×3).
    Returns:
        Filtered image as uint8.
    """
    kernel = avg_kernel(kernel_size)
    # Calling the convolution function to apply the filter to the image.
    result = MainConv(image.astype(np.float64), kernel)
    # Clipping to [0, 255] and converting back to uint8
    return np.clip(result, 0, 255).astype(np.uint8)


def gaussian_smoothing_filter(image: np.ndarray, kernel_size: int, variance: float) -> np.ndarray:
    """
    Apply a Gaussian smoothing filter.
    Arguments:
        image:       Input uint8 image.
        kernel_size: Size of the square kernel (must be odd).
        variance:    Gaussian variance.

    Returns:
        Filtered image as uint8.
    """
    kernel = gauss_kernel(kernel_size, variance)
    result = MainConv(image.astype(np.float64), kernel)
    return np.clip(result, 0, 255).astype(np.uint8)
