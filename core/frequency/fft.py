import numpy as np


def compute_fft(image):
    """Compute 2D FFT on the image.

    Returns
    -------
    F_shifted : ndarray complex128  — centered complex FFT (shared state for notch filtering)
    mag_display : ndarray uint8     — log-scaled magnitude, normalised to [0, 255] for display
    """
    if image.ndim == 3:
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    else:
        gray = image.astype(np.float64)

    F_shifted = np.fft.fftshift(np.fft.fft2(gray))

    magnitude = np.log(1.0 + np.abs(F_shifted))
    lo, hi = magnitude.min(), magnitude.max()
    if hi > lo:
        mag_display = ((magnitude - lo) / (hi - lo) * 255).astype(np.uint8)
    else:
        mag_display = np.zeros_like(magnitude, dtype=np.uint8)

    return F_shifted, mag_display
