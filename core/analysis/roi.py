import numpy as np


def extract_roi(image, x0, y0, x1, y1): #coordinates of the first and second click
    row0, row1 = min(y0, y1), max(y0, y1) #normalize to be between y coordinates of clicks (top-left corner)
    col0, col1 = min(x0, x1), max(x0, x1) #normalize to be between x coordinates of clicks (bottom-right corner)
    height, width = image.shape[:2] #height and width of the image
    row0, col0 = max(0, row0), max(0, col0)  #if user clicks outside the image set it to 0
    row1, col1 = min(height, row1), min(width, col1)   # if user clicks outside the image set it to max height or width

    if row1 <= row0 or col1 <= col0: 
        return None

    patch = image[row0:row1, col0:col1] #extract the patch of the image that is the ROI
    if patch.ndim == 3: #if image is RBG change is to grayscale
        patch = (0.299 * patch[..., 0] + 0.587 * patch[..., 1] + 0.114 * patch[..., 2])
    return np.clip(patch, 0, 255).astype(np.uint8) #clip values to be between 0, 255
