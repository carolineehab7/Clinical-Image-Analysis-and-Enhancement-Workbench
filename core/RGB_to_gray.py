import numpy as np

# Convert RGB to grayscale
def RGB_to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        # using the ITU-R BT.601 standard: weights: 0.299 R + 0.587 G + 0.114 B
        return (0.299 * image[:, :, 0] +
                0.587 * image[:, :, 1] +
                0.114 * image[:, :, 2])
    return image.astype(np.float64)