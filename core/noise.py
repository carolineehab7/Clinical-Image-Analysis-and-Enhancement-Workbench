import numpy as np


def _as_float_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float64, copy=True)


def _clip_to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 10.0,
                       seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = _as_float_image(image) + rng.normal(mean, sigma, size=image.shape)
    return _clip_to_uint8(noisy)


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.05,
                          salt_vs_pepper: float = 0.5,
                          seed: int | None = None) -> np.ndarray:
    if not 0 <= amount <= 1:
        raise ValueError("amount must be between 0 and 1")
    if not 0 <= salt_vs_pepper <= 1:
        raise ValueError("salt_vs_pepper must be between 0 and 1")

    rng = np.random.default_rng(seed)
    noisy = image.copy()
    total_pixels = image.shape[0] * image.shape[1]
    num_salt = int(np.ceil(total_pixels * amount * salt_vs_pepper))
    num_pepper = int(np.ceil(total_pixels * amount * (1.0 - salt_vs_pepper)))

    salt_rows = rng.integers(0, image.shape[0], size=num_salt)
    salt_cols = rng.integers(0, image.shape[1], size=num_salt)
    pepper_rows = rng.integers(0, image.shape[0], size=num_pepper)
    pepper_cols = rng.integers(0, image.shape[1], size=num_pepper)

    if image.ndim == 2:
        noisy[salt_rows, salt_cols] = 255
        noisy[pepper_rows, pepper_cols] = 0
    elif image.ndim == 3:
        noisy[salt_rows, salt_cols, :] = 255
        noisy[pepper_rows, pepper_cols, :] = 0
    else:
        raise ValueError("Unsupported image shape for salt-and-pepper noise")

    return noisy.astype(np.uint8)


def add_speckle_noise(image: np.ndarray, sigma: float = 0.1,
                      seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = _as_float_image(image)
    noise = rng.normal(0.0, sigma, size=image.shape)
    noisy = base + base * noise
    return _clip_to_uint8(noisy)


def add_poisson_noise(image: np.ndarray, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = _as_float_image(image)
    vals = np.clip(base, 0, 255)
    noisy = rng.poisson(vals).astype(np.float64)
    return _clip_to_uint8(noisy)


def add_uniform_noise(image: np.ndarray, low: float = -20.0, high: float = 20.0,
                      seed: int | None = None) -> np.ndarray:
    if high < low:
        raise ValueError("high must be greater than or equal to low")

    rng = np.random.default_rng(seed)
    noisy = _as_float_image(image) + rng.uniform(low, high, size=image.shape)
    return _clip_to_uint8(noisy)