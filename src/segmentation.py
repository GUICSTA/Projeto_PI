
from collections import deque

import numpy as np

from . import preprocessing

PAPER_MIN_BRIGHTNESS = 115
CARD_SCALE = 4
FULL_CARD_RATIO = 0.9
CORNER_WIDTH_RATIO = 0.28
CORNER_HEIGHT_RATIO = 0.45
CORNER_PAD_RATIO = 0.06
EDGE_MARGIN_RATIO = 0.015
MIN_ELEMENT_AREA_RATIO = 0.002
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def bounding_box(binary_array):
    """Retorna (y_min, y_max, x_min, x_max) da região não-zero (símbolo)."""
    ys, xs = np.where(binary_array > 0)
    if len(ys) == 0:
        raise ValueError("Nenhum pixel de símbolo encontrado na imagem binarizada.")
    return ys.min(), ys.max(), xs.min(), xs.max()


def crop_symbol(rgb_array, binary_array, margin=5):
    """Recorta a região do símbolo (com margem) tanto no RGB quanto na máscara binária."""
    y_min, y_max, x_min, x_max = bounding_box(binary_array)
    h, w = binary_array.shape
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(h, y_max + margin)
    x_max = min(w, x_max + margin)

    rgb_crop = rgb_array[y_min:y_max, x_min:x_max]
    mask_crop = binary_array[y_min:y_max, x_min:x_max]
    return rgb_crop, mask_crop


def dominant_color(rgb_array, mask):
    symbol_pixels = rgb_array[mask > 0]
    if len(symbol_pixels) == 0:
        raise ValueError("Máscara vazia — nenhum pixel de símbolo para analisar cor.")

    mean_r = symbol_pixels[:, 0].mean()
    mean_g = symbol_pixels[:, 1].mean()
    mean_b = symbol_pixels[:, 2].mean()

    # Vermelho: canal R significativamente maior que G e B.
    if mean_r - ((mean_g + mean_b) / 2) > 15:
        return "vermelho"
    return "preto"


def connected_components(binary_array):
    foreground = (binary_array > 0).tolist()
    h, w = binary_array.shape
    labels = [[0] * w for _ in range(h)]
    components = []

    for y_start in range(h):
        for x_start in range(w):
            if not foreground[y_start][x_start] or labels[y_start][x_start]:
                continue
            label = len(components) + 1
            labels[y_start][x_start] = label
            queue = deque([(y_start, x_start)])
            area, y_min, y_max, x_min, x_max = 0, y_start, y_start, x_start, x_start
            while queue:
                y, x = queue.popleft()
                area += 1
                y_min, y_max = min(y_min, y), max(y_max, y)
                x_min, x_max = min(x_min, x), max(x_max, x)
                for dy, dx in NEIGHBORS:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and foreground[ny][nx] and not labels[ny][nx]:
                        labels[ny][nx] = label
                        queue.append((ny, nx))
            components.append({"label": label, "area": area, "bbox": (y_min, y_max, x_min, x_max)})

    return np.array(labels, dtype=np.int32).reshape(h, w), components


def erode(mask, radius):
    result = mask.copy()
    for _ in range(radius):
        shrunk = result.copy()
        shrunk[1:] &= result[:-1]
        shrunk[:-1] &= result[1:]
        shrunk[:, 1:] &= result[:, :-1]
        shrunk[:, :-1] &= result[:, 1:]
        result = shrunk
    return result


def dilate(mask, radius):
    result = mask.copy()
    for _ in range(radius):
        grown = result.copy()
        grown[1:] |= result[:-1]
        grown[:-1] |= result[1:]
        grown[:, 1:] |= result[:, :-1]
        grown[:, :-1] |= result[:, 1:]
        result = grown
    return result


def opening(mask, radius):
    return dilate(erode(mask, radius), radius)


def closing(mask, radius):
    return erode(dilate(mask, radius), radius)


MORPHOLOGY = {"abertura": opening, "fechamento": closing}


def card_mask(rgb_array, scale=CARD_SCALE):
    h, w = rgb_array.shape[:2]
    small = rgb_array[::scale, ::scale]
    paper = small.min(axis=2) >= PAPER_MIN_BRIGHTNESS

    labels, components = connected_components(paper)
    if not components:
        raise ValueError("Nenhuma região de carta encontrada na imagem.")
    largest = max(components, key=lambda c: c["area"])["label"]
    card = labels == largest

    outside_labels, _ = connected_components(~card)
    border = np.concatenate(
        [outside_labels[0], outside_labels[-1], outside_labels[:, 0], outside_labels[:, -1]]
    )
    border_labels = np.setdiff1d(np.unique(border), [0])
    filled = ~np.isin(outside_labels, border_labels)

    return np.repeat(np.repeat(filled, scale, axis=0), scale, axis=1)[:h, :w]


def has_background(card):
    return card.mean() < FULL_CARD_RATIO


def corner_region(rgb_array, card):
    h, w = card.shape
    ys, xs = np.nonzero(card)
    card_h = ys.max() - ys.min() + 1
    card_w = xs.max() - xs.min() + 1
    corner = np.argmin(ys + xs)
    y_corner, x_corner = ys[corner], xs[corner]

    y0 = max(0, y_corner - int(card_h * CORNER_PAD_RATIO))
    x0 = max(0, x_corner - int(card_w * CORNER_PAD_RATIO))
    y1 = min(h, y_corner + int(card_h * CORNER_HEIGHT_RATIO))
    x1 = min(w, x_corner + int(card_w * CORNER_WIDTH_RATIO))

    margin = max(1, int(round(min(card_h, card_w) * EDGE_MARGIN_RATIO)))
    inside = erode(card, margin)[y0:y1, x0:x1]
    return rgb_array[y0:y1, x0:x1], inside, (y0, y1, x0, x1)


def _touches_far_border(bbox, shape):
    return bbox[1] >= shape[0] - 1 or bbox[3] >= shape[1] - 1


def _vertical_overlap(a, b):
    return min(a[1], b[1]) - max(a[0], b[0]) >= 0


def _horizontal_overlap(a, b):
    return min(a[3], b[3]) - max(a[2], b[2]) >= 0


def _union(boxes):
    return (
        min(b[0] for b in boxes),
        max(b[1] for b in boxes),
        min(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _offset(bbox, y0, x0):
    return (bbox[0] + y0, bbox[1] + y0, bbox[2] + x0, bbox[3] + x0)


def corner_elements(rgb_crop, inside, morphology=None, radius=1):
    ink_level = np.where(inside, rgb_crop.min(axis=2), 255).astype(np.uint8)
    binary, threshold = preprocessing.binarize(ink_level)
    binary[~inside] = 0
    if morphology:
        binary = MORPHOLOGY[morphology](binary > 0, radius).astype(np.uint8) * 255

    labels, components = connected_components(binary)
    min_area = MIN_ELEMENT_AREA_RATIO * binary.size
    elements = [
        c for c in components
        if c["area"] >= min_area and not _touches_far_border(c["bbox"], binary.shape)
    ]
    return labels, elements, threshold


def split_value_and_symbol(elements):
    if not elements:
        raise ValueError("Nenhum elemento encontrado no canto da carta.")

    ordered = sorted(elements, key=lambda c: (c["bbox"][0], c["bbox"][2]))
    first = ordered[0]["bbox"]
    first_width = first[3] - first[2] + 1
    value = [
        c for c in ordered
        if _vertical_overlap(c["bbox"], first) and c["bbox"][2] <= first[3] + first_width
    ]
    value_box = _union([c["bbox"] for c in value])

    below = [
        c for c in ordered
        if c not in value
        and c["bbox"][0] > value_box[1]
        and _horizontal_overlap(c["bbox"], value_box)
    ]
    if not below:
        raise ValueError("Nenhum símbolo de naipe encontrado abaixo do valor.")
    symbol = min(below, key=lambda c: c["bbox"][0])
    return value, symbol


def segment_corner(rgb_array, card=None, margin=5, morphology=None, radius=1):
    if card is None:
        card = card_mask(rgb_array)
    rgb_crop, inside, (y0, y1, x0, x1) = corner_region(rgb_array, card)
    labels, elements, threshold = corner_elements(rgb_crop, inside, morphology, radius)
    value, symbol = split_value_and_symbol(elements)

    sy0, sy1, sx0, sx1 = symbol["bbox"]
    h, w = labels.shape
    sy0, sx0 = max(0, sy0 - margin), max(0, sx0 - margin)
    sy1, sx1 = min(h, sy1 + margin + 1), min(w, sx1 + margin + 1)
    symbol_mask = (labels[sy0:sy1, sx0:sx1] == symbol["label"]).astype(np.uint8) * 255

    return {
        "regiao": (y0, y1, x0, x1),
        "elementos": [_offset(c["bbox"], y0, x0) for c in elements],
        "valor": _offset(_union([c["bbox"] for c in value]), y0, x0),
        "naipe": _offset(symbol["bbox"], y0, x0),
        "simbolo_rgb": rgb_crop[sy0:sy1, sx0:sx1],
        "simbolo_mascara": symbol_mask,
        "threshold": threshold,
    }
