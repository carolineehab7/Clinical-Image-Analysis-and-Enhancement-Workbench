import numpy as np
from .RGB_to_gray import RGB_to_gray
from core.convolution import _convolve_single
from .normalization import normalize

def sobel_filter(image: np.ndarray):
    gray = RGB_to_gray.RGB_to_gray.RGB_to_gray(image)

    # vertical sobel kernel
    Ky = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    # horizontal sobel kernel
    Kx = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float64)

    #gradient magnitude = sqrt(gx² + gy²)
    gx = _convolve_single(gray, Kx)
    gy = _convolve_single(gray, Ky)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    return normalize(gx), normalize(gy), normalize(magnitude)