"""
Convolution Implemented entirely from scratch.
Uses NumPy stride tricks to build an efficient sliding-window view,
then performs true mathematical convolution by flipping the kernel.
Boundary handling is done with 'reflect' padding to preserve edge information.
Used to apply the smoothing filters (average and Gaussian) to the image.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided

def MainConv(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a 2D convolution to a grayscale or RGB image.
    Strategy (stride tricks):
    1. Pad the image with 'reflect' boundary conditions to preserve edges.
    2. Flip the kernel horizontally and vertically for true convolution.
    3. Build a (H, W, kH, kW) view "as if is's 4D" of all overlapping patches using as_strided.
    4. Compute the dot product of each patch with the flipped kernel.

    Arguments:
        image:  Input array (H, W) or (H, W, C) for grayscale or RGB images.
        kernel: 2D filter kernel (kH, kW), float64.

    Returns:
        Convolved array as float64 (should then be clipped as needed).
    """
    if image.ndim == 3:
        # Process each RGB channel independently
        channels = [
            _core_single_conv(image[:, :, c].astype(np.float64), kernel)
            for c in range(image.shape[2]) # 0 red, 1 green, 2 blue.
        ]
        return np.stack(channels, axis=2)
    else:
        return _core_single_conv(image.astype(np.float64), kernel)


def _core_single_conv(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
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
    pad_top = (kh - 1) // 2
    pad_bottom = kh - 1 - pad_top
    pad_left = (kw - 1) // 2
    pad_right = kw - 1 - pad_left

    # Reflect padding
    padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='reflect'
    )

    # Build a sliding window view each patch is a neighnorhood (kh, kw) and we have (h, w).
    #the strides represent the value for moving one pixel at a time across the padded image, creating overlapping patches that correspond to the kernel size.
    shape = (h, w, kh, kw)
    strides = (
        padded.strides[0], #how many bytes to move down one row in the padded image
        padded.strides[1], #how many bytes to move right one column in the padded image
        padded.strides[0],
        padded.strides[1]
    )
    #creates a new view of the padded image without any loops.
    patches = as_strided(padded, shape=shape, strides=strides)

    # Element-wise multiply and sum
    result = np.einsum('ijkl,kl->ij', patches, kernel)

    return result
