"""
Convolution Module — Implemented entirely from scratch.
Uses NumPy stride tricks to build an efficient sliding-window view,
then performs true mathematical convolution by flipping the kernel.
No scipy.signal.convolve2d or any other convolution library is used.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a 2D convolution to a grayscale or RGB image.

    Strategy (stride tricks):
    1. Pad the image with 'reflect' boundary conditions to preserve edges.
    2. Flip the kernel horizontally and vertically for true convolution.
    3. Build a (H, W, kH, kW) view of all overlapping patches using as_strided.
    4. Compute the dot product of each patch with the flipped kernel.

    Args:
        image:  Input array (H, W) or (H, W, C), any numeric dtype.
        kernel: 2D filter kernel (kH, kW), float64.

    Returns:
        Convolved array as float64 (caller should clip/cast as needed).
    """
    if image.ndim == 3:
        # Process each RGB channel independently
        channels = [
            _convolve_single(image[:, :, c].astype(np.float64), kernel)
            for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=2)
    else:
        return _convolve_single(image.astype(np.float64), kernel)


def _convolve_single(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a single-channel (2D) image with a kernel.

    Boundary handling:
        Reflect padding — mirrors pixels at borders.

    True convolution:
        Kernel is flipped both vertically and horizontally
        before multiplication.
    """
    # Flip kernel for true mathematical convolution
    kernel = np.flip(kernel, axis=(0, 1))

    h, w = image.shape
    kh, kw = kernel.shape

    # Compute padding size
    pad_h = kh // 2
    pad_w = kw // 2

    # Reflect padding
    padded = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode='reflect'
    )

    # Build sliding window view
    shape = (h, w, kh, kw)
    strides = (
        padded.strides[0],
        padded.strides[1],
        padded.strides[0],
        padded.strides[1]
    )

    patches = as_strided(padded, shape=shape, strides=strides)

    # Element-wise multiply and sum
    result = np.einsum('ijkl,kl->ij', patches, kernel)

    return result