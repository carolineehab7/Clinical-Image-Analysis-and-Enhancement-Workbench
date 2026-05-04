"""
Convolution Module — Implemented entirely from scratch.
Uses NumPy stride tricks to build an efficient sliding-window view,
then performs element-wise multiplication with the kernel.
No scipy.signal.convolve2d or any other convolution library is used.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a 2D convolution to a grayscale or RGB image.

    Strategy (stride tricks):
    1. Pad the image with 'reflect' boundary conditions to preserve edges.
    2. Build a (H, W, kH, kW) view of all overlapping patches using as_strided.
    3. Compute the dot product of each patch with the kernel via einsum.

    This is equivalent to a nested-loop convolution but vectorised for speed.
    The mathematical operation is identical — no built-in filter is used.

    Args:
        image:  Input array (H, W) or (H, W, C), any numeric dtype.
        kernel: 2D filter kernel (kH, kW), float64.

    Returns:
        Convolved array as float64 (caller should clip/cast as needed).
    """
    if image.ndim == 3:
        channels = [_convolve_single(image[:, :, c].astype(np.float64), kernel)
                    for c in range(image.shape[2])]
        return np.stack(channels, axis=2)
    else:
        return _convolve_single(image.astype(np.float64), kernel)


def _convolve_single(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a single-channel (2D) image with a kernel.

    Boundary handling: reflect — mirror pixels at borders so edge results
    are consistent with an interior convolution.
    """
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    ph, pw = padded.shape
    shape = (h, w, kh, kw)
    s = padded.strides
    strides = (s[0], s[1], s[0], s[1])

    patches = as_strided(padded, shape=shape, strides=strides)

    result = np.einsum('ijkl,kl->ij', patches, kernel)
    return result
