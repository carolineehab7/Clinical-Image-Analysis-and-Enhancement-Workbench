import numpy as np


def _freq_grids(shape):
    """Return (U, V) meshgrids centred at (0, 0) for a given FFT shape."""
    rows, cols = shape
    U, V = np.meshgrid(
        np.arange(rows) - rows // 2,
        np.arange(cols) - cols // 2,
        indexing="ij",
    )
    return U, V


def ideal_notch_mask(shape, notches, D0):
    """Binary notch-reject mask: 0 inside radius D0 at each notch + conjugate."""
    U, V = _freq_grids(shape)
    mask = np.ones(shape, dtype=np.float64)
    for (u0, v0) in notches:
        mask[np.sqrt((U - u0) ** 2 + (V - v0) ** 2) <= D0] = 0.0
        mask[np.sqrt((U + u0) ** 2 + (V + v0) ** 2) <= D0] = 0.0
    return mask


def butterworth_notch_mask(shape, notches, D0, n):
    """Butterworth notch-reject mask of order n."""
    U, V = _freq_grids(shape)
    mask = np.ones(shape, dtype=np.float64)
    for (u0, v0) in notches:
        D1 = np.sqrt((U - u0) ** 2 + (V - v0) ** 2) + 1e-10
        D2 = np.sqrt((U + u0) ** 2 + (V + v0) ** 2) + 1e-10
        mask *= (1.0 / (1.0 + (D0 / D1) ** (2 * n))) * (1.0 / (1.0 + (D0 / D2) ** (2 * n)))
    return mask


def gaussian_notch_mask(shape, notches, D0):
    """Gaussian notch-reject mask."""
    U, V = _freq_grids(shape)
    mask = np.ones(shape, dtype=np.float64)
    two_D0_sq = max(2.0 * D0 ** 2, 1e-10)
    for (u0, v0) in notches:
        mask *= (1.0 - np.exp(-((U - u0) ** 2 + (V - v0) ** 2) / two_D0_sq))
        mask *= (1.0 - np.exp(-((U + u0) ** 2 + (V + v0) ** 2) / two_D0_sq))
    return mask


def apply_notch_filter(F_shifted, mask):
    """Multiply mask × FFT → IFFT → real → clip → uint8."""
    result = np.real(np.fft.ifft2(np.fft.ifftshift(F_shifted * mask)))
    return np.clip(result, 0, 255).astype(np.uint8)
