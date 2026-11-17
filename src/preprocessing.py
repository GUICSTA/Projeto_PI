
import numpy as np
from PIL import Image


def load_rgb(path):
    """Carrega a imagem e garante RGB explicitamente (lição aprendida no M1.1)."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def to_grayscale(rgb_array):
    """Conversão manual para escala de cinza (luminância ponderada)."""
    r = rgb_array[:, :, 0].astype(np.float64)
    g = rgb_array[:, :, 1].astype(np.float64)
    b = rgb_array[:, :, 2].astype(np.float64)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)


def binarize(gray_array, threshold=None):
    if threshold is None:
        threshold = _otsu_threshold(gray_array)
    binary = (gray_array <= threshold).astype(np.uint8) * 255
    return binary, threshold


def _otsu_threshold(gray_array):
    """Implementação manual do método de Otsu (nível de pixel, sem lib pronta)."""
    hist, _ = np.histogram(gray_array, bins=256, range=(0, 256))
    total = gray_array.size
    sum_total = np.dot(np.arange(256), hist)

    sum_bg, weight_bg = 0.0, 0.0
    max_between, best_thresh = 0.0, 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_between:
            max_between = between
            best_thresh = t

    return best_thresh


def median_filter(gray_array, radius):
    """Filtro de mediana com janela quadrada (2*radius+1)², borda replicada."""
    if radius <= 0:
        return gray_array
    size = 2 * radius + 1
    padded = np.pad(gray_array, radius, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    return np.median(windows, axis=(2, 3)).astype(gray_array.dtype)
