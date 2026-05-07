import numpy as np
from .convolution import _core_single_conv
from ..enhancement.color_conversion import RGB_to_gray
from ..enhancement.normalization import normalize

def sobel_filter(image: np.ndarray):
    gray = RGB_to_gray(image)

    # vertical sobel kernel
    Ky = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    # horizontal sobel kernel
    Kx = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float64)

    #gradient magnitude = sqrt(gx² + gy²)
    gx = _core_single_conv(gray, Kx)
    gy = _core_single_conv(gray, Ky)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    return normalize(gx), normalize(gy), normalize(magnitude)
