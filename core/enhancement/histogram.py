import numpy as np
from .color_conversion import RGB_to_gray

# Computes the histogram of a grayscale image
def compute_histogram(image: np.ndarray) -> np.ndarray:
    gray = _ensure_gray(image).flatten().astype(np.uint8)
    hist = np.zeros(256, dtype=np.int64)
    for intensity in range(256):
        hist[intensity] = np.sum(gray == intensity)
    return hist


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    return RGB_to_gray(image)

#converts the 2D grid of pixels into a 1D list. A 512×512 image becomes a list of 262,144 numbers
def _equalize_block(block: np.ndarray) -> np.ndarray:
    flat = block.flatten().astype(np.uint8)
    n_pixels = flat.size
    hist = np.zeros(256, dtype=np.int64)
    for intensity in range(256):
        hist[intensity] = int(np.sum(flat == intensity))
#compute the cumulative distribution function (CDF) from the histogram, which is used to map the original pixel values to their new equalized values
    cdf = np.zeros(256, dtype=np.int64)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]
# find the minimum non-zero value in the CDF, which is used to ensure that the mapping starts from the lowest intensity level that actually appears in the block. This prevents issues with blocks that have a lot of zero-intensity pixels (e.g., very dark areas) and ensures a more effective equalization
    cdf_min = 0
    for val in cdf:
        if val > 0:
            cdf_min = int(val)
            break

    denominator = n_pixels - cdf_min

    if denominator <= 0:
        return block.astype(np.uint8).copy()
# create a lookup table (LUT) to map each original pixel intensity to its new equalized value based on the CDF
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        val = int(round((cdf[i] - cdf_min) / denominator * 255))
        lut[i] = max(0, min(255, val))

    return lut[block.astype(np.uint8)]

# Applies local histogram equalization to the input image by dividing it into blocks of a specified size and equalizing each block independently
def local_histogram_equalization(image: np.ndarray, block_size: int) -> np.ndarray:
    gray = _ensure_gray(image)
    h, w = gray.shape
    result = np.zeros_like(gray, dtype=np.uint8)

# The function iterates over the image in steps of block_size, extracting each block and applying the _equalize_block function to it
    for row_start in range(0, h, block_size):
        for col_start in range(0, w, block_size):
            row_end = min(row_start + block_size, h)
            col_end = min(col_start + block_size, w)
# The block is extracted from the grayscale image, equalized using the _equalize_block function, and then placed back into the corresponding location in the result image
            block = gray[row_start:row_end, col_start:col_end]
            eq_block = _equalize_block(block)
            result[row_start:row_end, col_start:col_end] = eq_block

    return result
