import numpy as np
from numpy.lib.stride_tricks import as_strided

def median_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    if image.ndim == 3: #if the image is RGB, we should apply the median filter to each channel separately
        channels = [median_single(image[:, :, c], kernel_size)
                    for c in range(image.shape[2])]
        return np.stack(channels, axis=2) #stack channels back together
    return median_single(image, kernel_size)

#apply the median filter to a single 2D channel
def median_single(image: np.ndarray, kernel_size: int) -> np.ndarray:

    height, width = image.shape
    pad = kernel_size // 2

    # reflect borders to avoid dark borders in the output
    padded = np.pad(image.astype(np.float64), pad, mode='reflect')

    # create a window view (Height, Width, kernel Size, kernel Size)
    shape = (height, width, kernel_size, kernel_size)
    s = padded.strides
    strides = (s[0], s[1], s[0], s[1])  
    patches = as_strided(padded, shape=shape, strides=strides)
    
    n = kernel_size * kernel_size
    mid_idx = n // 2 # get the median index
    flat_patches = patches.reshape(height, width, n)

    # Sort each patch alone then combine them
    sorted_patches = np.sort(flat_patches, axis=2)
    result = sorted_patches[:, :, mid_idx] # take the median value from each patch

    return np.clip(result, 0, 255).astype(np.uint8)
