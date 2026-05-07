import numpy as np

def nearest_neighbor_zoom(image: np.ndarray, scale: float) -> np.ndarray:

    if scale <= 0:
        raise ValueError("Scale must be positive.")

    height, width = image.shape[:2]

    # calculate the new dimensions of output image, gets the max between 1 
    # (least number 1x1) and the scaled dimension to nearest integer
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))

    destination_rows = np.arange(new_height) #create 1D array of this new height (starts from 0 to new_height-1)
    destination_cols = np.arange(new_width) #create 1D array of this new width (starts from 0 to new_width-1)

    #overlay the new grid on the original image
    src_rows = destination_rows / scale
    src_cols = destination_cols / scale

    # find the closest original pixel for each new pixel 
    # copies the intensity of the nearest pixel of the original to the new pixel
    src_row_idx = np.clip(np.round(src_rows).astype(np.int64), 0, height - 1)
    src_col_idx = np.clip(np.round(src_cols).astype(np.int64), 0, width - 1)

    result = image[np.ix_(src_row_idx, src_col_idx)] # creates 2D array with zoomed image

    return result.astype(np.uint8)


def bilinear_zoom(image: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("Scale must be positive.")

    height, width = image.shape[:2]
    # calculate the new dimensions of output image, gets the max between 1 
    # (least number 1x1) and the scaled dimension to nearest integer
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))

    destination_rows = np.arange(new_height, dtype=np.float64)
    destination_cols = np.arange(new_width, dtype=np.float64)

    src_rows = destination_rows / scale  
    src_cols = destination_cols / scale  

    # floors top-left corner (i = r0,j = c0)
    row0 = np.floor(src_rows).astype(np.int64) 
    col0 = np.floor(src_cols).astype(np.int64)

    # get bottom right corner, +1 to top left corner
    # (i + 1, j + 1) --> (r1, c1)
    row1 = np.clip(row0 + 1, 0, height - 1)
    col1 = np.clip(col0 + 1, 0, width - 1)
    row0 = np.clip(row0, 0, height - 1)
    col0 = np.clip(col0, 0, width - 1)
    
    # fractional distance for rows and columns
    x = (src_rows - np.floor(src_rows)).reshape(new_height, 1) 
    y = (src_cols - np.floor(src_cols)).reshape(1, new_width)  

    def _interp_channel(imagechannel: np.ndarray) -> np.ndarray:

        top_left = imagechannel[row0][:, col0].astype(np.float64)   # Top-left corner
        top_right = imagechannel[row0][:, col1].astype(np.float64)   # Top-right corner
        bottom_left = imagechannel[row1][:, col0].astype(np.float64)   # Bottom-left corner
        bottom_right = imagechannel[row1][:, col1].astype(np.float64)   # Bottom-right corner

        # bilinear interpolation formula
        # v(x,y) = a(1-x)(1-y) + b(1-x)y + cx(1-y) + dxy
        #Coefficients a,b,c,d are determined from the four nearest neighbors of (x,y)
        interpolated = (top_left * (1.0 - x) * (1.0 - y) +
                        top_right * (1.0 - x) * y +
                        bottom_left * x * (1.0 - y) +
                        bottom_right * x * y)
        return np.clip(interpolated, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return _interp_channel(image) #2D grayscale image
    else:
        channels = [_interp_channel(image[:, :, i]) for i in range(image.shape[2])]
        return np.stack(channels, axis=2) #3D RGB image
