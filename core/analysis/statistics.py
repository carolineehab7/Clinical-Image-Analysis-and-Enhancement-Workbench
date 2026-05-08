def compute_local_histogram(roi):
    """Return a 256-bin histogram for the ROI.

    Reuses Phase 1's compute_histogram, which loops over intensity levels
    from scratch without np.histogram. Converted to a plain list so the
    rest of Member 4's code is unchanged.
    """
    from core.enhancement.histogram import compute_histogram
    return list(compute_histogram(roi))


def compute_mean(roi):
    """Arithmetic mean computed directly from pixel values."""
    flat = roi.ravel()
    n = len(flat)
    if n == 0:
        return 0.0
    total = 0.0
    for px in flat:
        total += float(px)
    return total / n


def compute_variance(roi):
    """Population variance computed directly from pixel values."""
    flat = roi.ravel()
    n = len(flat)
    if n == 0:
        return 0.0
    mean = compute_mean(roi)
    sq = 0.0
    for px in flat:
        diff = float(px) - mean
        sq += diff * diff
    return sq / n
