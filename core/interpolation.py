"""
Interpolation Module — Implemented entirely from scratch.
Provides Nearest-Neighbor and Bilinear interpolation for image zoom.
No built-in zoom or resize libraries are used.
"""

import numpy as np


def nearest_neighbor_zoom(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Resize an image using Nearest-Neighbor interpolation.

    For each destination pixel (r_dst, c_dst), we map back to source coordinates:
        r_src = r_dst / scale
        c_src = c_dst / scale
    Then we round to the nearest integer to pick the source pixel.

    Args:
        image: Input image array (H, W) grayscale or (H, W, C) color, uint8.
        scale: Zoom factor. >1 zooms in, <1 zooms out.

    Returns:
        Resized image as uint8 array.
    """
    if scale <= 0:
        raise ValueError("Scale must be positive.")

    h, w = image.shape[:2]
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # Compute source row and column indices for every destination pixel
    # Reverse mapping: dst -> src
    dst_rows = np.arange(new_h)
    dst_cols = np.arange(new_w)

    # Map destination pixel centers back to source space
    src_rows = dst_rows / scale
    src_cols = dst_cols / scale

    # Round to nearest integer (nearest neighbor selection)
    src_row_idx = np.clip(np.round(src_rows).astype(np.int64), 0, h - 1)
    src_col_idx = np.clip(np.round(src_cols).astype(np.int64), 0, w - 1)

    # Advanced indexing to build output — works for 2D and 3D
    if image.ndim == 2:
        result = image[np.ix_(src_row_idx, src_col_idx)]
    else:
        # For each channel, apply the same index mapping
        result = image[np.ix_(src_row_idx, src_col_idx)]  # numpy handles 3D ix_

    return result.astype(np.uint8)


def bilinear_zoom(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Resize an image using Bilinear interpolation.

    For each destination pixel, we compute its fractional source coordinates,
    identify the four surrounding source pixels, and interpolate:

        P = (1-dr)(1-dc)*TL + (1-dr)*dc*TR + dr*(1-dc)*BL + dr*dc*BR

    where dr, dc are the fractional parts of the source row/column.

    Args:
        image: Input image array (H, W) or (H, W, C), uint8.
        scale: Zoom factor.

    Returns:
        Resized image as uint8 array.
    """
    if scale <= 0:
        raise ValueError("Scale must be positive.")

    h, w = image.shape[:2]
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # Reverse mapping: for each destination pixel, find source coordinates
    dst_rows = np.arange(new_h, dtype=np.float64)
    dst_cols = np.arange(new_w, dtype=np.float64)

    src_rows = dst_rows / scale  # Shape: (new_h,)
    src_cols = dst_cols / scale  # Shape: (new_w,)

    # Floor = top-left corner of surrounding square
    r0 = np.floor(src_rows).astype(np.int64)
    c0 = np.floor(src_cols).astype(np.int64)

    # Bottom-right corner (clamped to image bounds)
    r1 = np.clip(r0 + 1, 0, h - 1)
    c1 = np.clip(c0 + 1, 0, w - 1)
    r0 = np.clip(r0, 0, h - 1)
    c0 = np.clip(c0, 0, w - 1)

    # Fractional distances from top-left corner
    dr = (src_rows - np.floor(src_rows)).reshape(new_h, 1)  # (new_h, 1) for broadcast
    dc = (src_cols - np.floor(src_cols)).reshape(1, new_w)  # (1, new_w) for broadcast

    def _interp_channel(ch: np.ndarray) -> np.ndarray:
        """Apply bilinear interpolation to a single 2D channel."""
        # Four corners of the interpolation square
        tl = ch[r0][:, c0].astype(np.float64)   # Top-left
        tr = ch[r0][:, c1].astype(np.float64)   # Top-right
        bl = ch[r1][:, c0].astype(np.float64)   # Bottom-left
        br = ch[r1][:, c1].astype(np.float64)   # Bottom-right

        # Bilinear formula
        interpolated = (tl * (1.0 - dr) * (1.0 - dc) +
                        tr * (1.0 - dr) * dc +
                        bl * dr * (1.0 - dc) +
                        br * dr * dc)
        return np.clip(interpolated, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return _interp_channel(image)
    else:
        # Apply per channel and stack back
        channels = [_interp_channel(image[:, :, c]) for c in range(image.shape[2])]
        return np.stack(channels, axis=2)
