import numpy as np
import RGB_to_gray as RBG_To_Gray
from core.convolution import _convolve_single
import normalization as Normalization

def sobel_filter(image: np.ndarray):
    gray = RBG_To_Gray.RGB_to_gray.RGB_to_gray(image)

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

    return Normalization(gx), Normalization(gy), Normalization(magnitude)