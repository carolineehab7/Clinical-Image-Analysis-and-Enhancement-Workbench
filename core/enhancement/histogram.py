import numpy as np
from .color_conversion import RGB_to_gray

def compute_histogram(flat: np.ndarray) -> np.ndarray:
    hist = np.zeros(256, dtype=np.int64)
    # Count pixels at each intensity level by comparing the whole array at once
    for intensity in range(256):
        hist[intensity] = np.sum(flat == intensity)
    return hist


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    return RGB_to_gray(image)



# Builds a 256-entry Look-Up Table (LUT) that maps each original intensity
# value to its histogram-equalised output value for a single image block.
def _compute_block_lut(block_flat: np.ndarray) -> np.ndarray:
    flat = block_flat.astype(np.uint8)
    # Total number of pixels in this block — used as the normalisation factor
    n = len(flat)

    # Stage 1: histogram of this block's pixel intensities
    hist = compute_histogram(flat)

    # cdf[i] = total number of pixels with intensity 0 … i
    cdf = np.zeros(256, dtype=np.int64)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    # Find the first non-zero CDF value (cdf_min)
    cdf_min = int(next((v for v in cdf if v > 0), 0))

    # Denominator = total pixels minus the cdf_min offset
    # A value of 0 means the block is completely uniform — skip equalisation
    denom = n - cdf_min

    # build the LUT using the standard histogram equalisation formula
    lut = np.zeros(256, dtype=np.float32)
    if denom > 0:
        for i in range(256):
            # Scale the CDF into [0, 255] and clip to avoid any floating-point overshoot
            lut[i] = float(np.clip(round((cdf[i] - cdf_min) / denom * 255), 0, 255))
    return lut



def local_histogram_equalization_interpolated(image: np.ndarray, block_size: int) -> np.ndarray:
    # Convert to uint8 grayscale 
    gray = _ensure_gray(image).astype(np.uint8)
    h, w = gray.shape

    # Number of blocks needed to cover the image (last block may be smaller)
    n_rows = max(1, (h + block_size - 1) // block_size)
    n_cols = max(1, (w + block_size - 1) // block_size)

    # compute one LUT per block 
    luts = np.zeros((n_rows, n_cols, 256), dtype=np.float32)
    for by in range(n_rows):
        for bx in range(n_cols):
            # Extract block boundaries, clamping the last block to image edges
            r0 = by * block_size;  r1 = min(r0 + block_size, h)
            c0 = bx * block_size;  c1 = min(c0 + block_size, w)
            # Flatten the block to a 1-D array and compute its LUT
            luts[by, bx] = _compute_block_lut(gray[r0:r1, c0:c1].ravel())

    # compute bilinear interpolation weights for every pixel
    # y_frac, x_frac is the fractional position of each pixel along the vertical and horizontal axes relative to the block grid.
    y_frac = np.arange(h, dtype=np.float32) / block_size - 0.5   # shape (h,)
    x_frac = np.arange(w, dtype=np.float32) / block_size - 0.5   # shape (w,)

    # Top/left block indices — clipped so border pixels don't go out of range
    by1 = np.clip(np.floor(y_frac).astype(int), 0, n_rows - 1)   # shape (h,)
    bx1 = np.clip(np.floor(x_frac).astype(int), 0, n_cols - 1)   # shape (w,)
    # Bottom/right block indices — clamped at the last block for edge pixels
    by2 = np.clip(by1 + 1, 0, n_rows - 1)
    bx2 = np.clip(bx1 + 1, 0, n_cols - 1)

    # Fractional weights: how far the pixel is from the top/left block centre
    # Reshaped for broadcasting: wy is (h,1), wx is (1,w)
    wy = np.clip(y_frac - np.floor(y_frac), 0.0, 1.0).astype(np.float32)[:, None]
    wx = np.clip(x_frac - np.floor(x_frac), 0.0, 1.0).astype(np.float32)[None, :]

    # ── Step 3: look up all four corner LUTs for every pixel at once ─────────
    # pix[r, c] = original intensity of pixel (r, c), used as the LUT index
    pix  = gray.astype(int)    # shape (h, w)

    # Reshape block indices for broadcasting against the (h, w) pixel grid
    by1_ = by1[:, None]        # shape (h, 1)
    bx1_ = bx1[None, :]       # shape (1, w)
    by2_ = by2[:, None]
    bx2_ = bx2[None, :]

    # Each vXY array is shape (h, w): the equalised intensity from that corner's LUT
    v00 = luts[by1_, bx1_, pix]   # top-left block LUT
    v01 = luts[by1_, bx2_, pix]   # top-right block LUT
    v10 = luts[by2_, bx1_, pix]   # bottom-left block LUT
    v11 = luts[by2_, bx2_, pix]   # bottom-right block LUT

    # Standard bilinear formula weighted by (wy, wx):
    # result = top-row blend * (1-wy)  +  bottom-row blend * wy
    # where each row blend is itself a linear mix along x using wx.
    result = ((1 - wy) * (1 - wx) * v00 +
              (1 - wy) *      wx  * v01 +
                   wy  * (1 - wx) * v10 +
                   wy  *      wx  * v11)

    # Clip any floating-point overshoot and convert back to uint8
    return np.clip(result, 0, 255).astype(np.uint8)
