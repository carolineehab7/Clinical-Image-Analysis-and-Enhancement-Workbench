import numpy as np
from .color_conversion import RGB_to_gray

# Computes the histogram of a grayscale image
def compute_histogram(flat: np.ndarray) -> np.ndarray:
    """Return a 256-bin histogram for a flattened uint8 array."""
    hist = np.zeros(256, dtype=np.int64)
    for intensity in range(256):
        hist[intensity] = np.sum(flat == intensity)
    return hist


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    return RGB_to_gray(image)


def _compute_block_lut(block_flat: np.ndarray) -> np.ndarray:
    """Return a float32 LUT[256] mapping original intensity → equalized intensity for one block."""
    flat = block_flat.astype(np.uint8)
    n = len(flat)
    hist = compute_histogram(flat)

    cdf = np.zeros(256, dtype=np.int64)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    cdf_min = int(next((v for v in cdf if v > 0), 0))
    denom = n - cdf_min

    lut = np.zeros(256, dtype=np.float32)
    if denom > 0:
        for i in range(256):
            lut[i] = float(np.clip(round((cdf[i] - cdf_min) / denom * 255), 0, 255))
    return lut



def local_histogram_equalization_interpolated(image: np.ndarray, block_size: int) -> np.ndarray:
    """Local HE with bilinear interpolation between block LUTs.

    Instead of assigning every pixel in a block a single equalization mapping,
    each pixel is the bilinear blend of the four nearest block-centre LUTs.
    This eliminates the hard block-boundary grid artifact.
    """
    gray = _ensure_gray(image).astype(np.uint8)
    h, w = gray.shape

    n_rows = max(1, (h + block_size - 1) // block_size)
    n_cols = max(1, (w + block_size - 1) // block_size)

    # ── Step 1: compute one LUT per block ──────────────────────────────
    luts = np.zeros((n_rows, n_cols, 256), dtype=np.float32)
    for by in range(n_rows):
        for bx in range(n_cols):
            r0 = by * block_size;  r1 = min(r0 + block_size, h)
            c0 = bx * block_size;  c1 = min(c0 + block_size, w)
            luts[by, bx] = _compute_block_lut(gray[r0:r1, c0:c1].ravel())

    # ── Step 2: bilinear interpolation weights for every pixel ─────────
    # Block (by, bx) is centred at row = (by + 0.5) * block_size.
    # Pixel at row r has fractional grid position  y = r/block_size - 0.5.
    y_frac = np.arange(h, dtype=np.float32) / block_size - 0.5   # (h,)
    x_frac = np.arange(w, dtype=np.float32) / block_size - 0.5   # (w,)

    by1 = np.clip(np.floor(y_frac).astype(int), 0, n_rows - 1)   # (h,)
    bx1 = np.clip(np.floor(x_frac).astype(int), 0, n_cols - 1)   # (w,)
    by2 = np.clip(by1 + 1, 0, n_rows - 1)
    bx2 = np.clip(bx1 + 1, 0, n_cols - 1)

    wy = np.clip(y_frac - np.floor(y_frac), 0.0, 1.0).astype(np.float32)[:, None]  # (h,1)
    wx = np.clip(x_frac - np.floor(x_frac), 0.0, 1.0).astype(np.float32)[None, :]  # (1,w)

    # ── Step 3: fancy-index all four corner LUTs at once ───────────────
    pix  = gray.astype(int)          # (h, w)  pixel intensities as indices
    by1_ = by1[:, None]              # (h, 1)
    bx1_ = bx1[None, :]             # (1, w)
    by2_ = by2[:, None]
    bx2_ = bx2[None, :]

    v00 = luts[by1_, bx1_, pix]     # top-left     block LUT applied to each pixel
    v01 = luts[by1_, bx2_, pix]     # top-right
    v10 = luts[by2_, bx1_, pix]     # bottom-left
    v11 = luts[by2_, bx2_, pix]     # bottom-right

    # ── Step 4: bilinear blend ─────────────────────────────────────────
    result = ((1 - wy) * (1 - wx) * v00 +
              (1 - wy) *      wx  * v01 +
                   wy  * (1 - wx) * v10 +
                   wy  *      wx  * v11)

    return np.clip(result, 0, 255).astype(np.uint8)
