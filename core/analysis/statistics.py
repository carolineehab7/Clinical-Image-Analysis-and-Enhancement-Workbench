def compute_local_histogram(roi):
    """Count pixels per intensity level [0..255] without np.histogram."""
    bins = [0] * 256
    for px in roi.ravel():
        v = int(px)
        if 0 <= v <= 255:
            bins[v] += 1
    return bins


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
