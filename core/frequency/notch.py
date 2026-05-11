import numpy as np


def _freq_grids(shape):
    rows, cols = shape
    U, V = np.meshgrid(
        #Centered frequency coordinates: range(-N//2, N//2) for each dimension.
        #gives every pixel in the spectrum a label (U, V) telling you its signed frequency coordinate, with (0, 0) at the center.
        np.arange(rows) - rows // 2,
        np.arange(cols) - cols // 2,
        indexing="ij",
    )
    return U, V


def ideal_notch_mask(shape, notches, D0):
    # Ideal notch-reject mask: 0 inside radius D0 of each notch, 1 elsewhere.
    # so the pixels in the FFT that are within D0 of any notch get zeroed out (removed), while the rest remain unchanged (multiplied by 1).
    U, V = _freq_grids(shape)
    mask = np.ones(shape, dtype=np.float64)
    for (u0, v0) in notches:
        mask[np.sqrt((U - u0) ** 2 + (V - v0) ** 2) <= D0] = 0.0
        mask[np.sqrt((U + u0) ** 2 + (V + v0) ** 2) <= D0] = 0.0
    return mask


def butterworth_notch_mask(shape, notches, D0, n):
  # Butterworth notch-reject mask: smooth transition from 1 to 0 around each notch, controlled by order n.
  # The mask value at each pixel is the product of the contributions from all notches, where each contribution is a smooth function that approaches 0 near the notch and 1 far away. Higher n means a sharper transition.
  # we multiply not add because we want the combined effect of all notches to be that a pixel is significantly attenuated (mask value near 0) if it is close to any notch, while remaining mostly unaffected (mask value near 1) if it is far from all notches.
  # By multiplying, we ensure the mask values stay in the range [0, 1], where 0 means complete rejection and 1 means no attenuation.
    U, V = _freq_grids(shape)
    mask = np.ones(shape, dtype=np.float64)
    for (u0, v0) in notches:
        D1 = np.sqrt((U - u0) ** 2 + (V - v0) ** 2) + 1e-10
        D2 = np.sqrt((U + u0) ** 2 + (V + v0) ** 2) + 1e-10
        mask *= (1.0 / (1.0 + (D0 / D1) ** (2 * n))) * (1.0 / (1.0 + (D0 / D2) ** (2 * n))) 
    return mask


def gaussian_notch_mask(shape, notches, D0):
   # Gaussian notch-reject mask: smooth exponential decay around each notch, controlled by D0.
   # Similar to Butterworth but with an exponential decay, which can provide a smoother transition
    U, V = _freq_grids(shape)
    mask = np.ones(shape, dtype=np.float64)
    two_D0_sq = max(2.0 * D0 ** 2, 1e-10) # Avoid division by zero if D0 is very small.
    for (u0, v0) in notches:
        mask *= (1.0 - np.exp(-((U - u0) ** 2 + (V - v0) ** 2) / two_D0_sq))
        mask *= (1.0 - np.exp(-((U + u0) ** 2 + (V + v0) ** 2) / two_D0_sq))
    return mask


def apply_notch_filter(F_shifted, mask):
    # Apply the notch filter by multiplying the FFT with the mask, then inverse FFT to get the filtered image.
    result = np.real(np.fft.ifft2(np.fft.ifftshift(F_shifted * mask)))
    return np.clip(result, 0, 255).astype(np.uint8)
