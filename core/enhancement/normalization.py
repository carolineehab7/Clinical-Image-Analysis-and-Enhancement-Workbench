import numpy as np

#Normalize
def normalize(arr: np.ndarray) -> np.ndarray:
    
    # it will get the min and the max values of the array
    min, max = arr.min(), arr.max()

    # normalize to [0,255] ((arr-min)/(max-min))*255
    if max > min:
        return ((arr - min) / (max - min) * 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)
