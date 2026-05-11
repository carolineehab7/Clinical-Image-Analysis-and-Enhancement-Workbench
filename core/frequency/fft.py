import numpy as np

def compute_fft(image: np.ndarray):
    """Compute a centered 2D FFT and a displayable magnitude spectrum.

    Parameters
    ----------
    image : np.ndarray
        Input image as a 2D grayscale array or a 3D RGB/RGBA array. Values are
        converted to float64 for FFT stability.

    Returns
    -------
    F_shifted : np.ndarray
        Complex128 array containing the centered FFT (fftshift(fft2)).
    mag_display : np.ndarray
        Uint8 log-magnitude image scaled to [0, 255] for display.

    Raises
    ------
    ValueError
        If the input is None, empty, not numeric, or has unsupported shape.
    """
    if image is None:
        raise ValueError("Image is None.")

    img = np.asarray(image)
    if img.size == 0:
        raise ValueError("Image is empty.")

    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        if img.shape[2] >= 3:
            # Use the first three channels; ignore alpha if present.
            gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        elif img.shape[2] == 1:
            gray = img[..., 0]
        else:
            raise ValueError("Unsupported channel count for image.")
    else:
        raise ValueError("Image must be 2D or 3D.")

    try:
        gray = gray.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Image must contain numeric values.") from exc

    # Guard against NaN/Inf so log scaling stays finite.
    gray = np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0)

    F_shifted = np.fft.fftshift(np.fft.fft2(gray))
    #spatial to freq then centers zero frequency.
    magnitude = np.log(1.0 + np.abs(F_shifted))
    # abs to get magniture from complex, log(1+|F|) for better visualization, scaled to [0,255]
    lo, hi = magnitude.min(), magnitude.max()
    if hi > lo: # to avoid division by zero if the image is constant
        mag_display = ((magnitude - lo) / (hi - lo) * 255).astype(np.uint8)
    else: #if the image is constant, the magnitude will be zero everywhere, so we can just return a zero array.
        mag_display = np.zeros_like(magnitude, dtype=np.uint8)
    # returns the complex FFT for filtering and inverse FFT, and the displayable magnitude spectrum as uint8
    return F_shifted, mag_display
