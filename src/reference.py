"""Máscaras de referência do símbolo do naipe e medida de Dice/IoU por pixel.

As referências são semi-automáticas e independentes da decisão do pipeline:
- caixa de busca = caixa do naipe na foto limpa ampliada em 20 px;
- limiar FIXO em min(R,G,B) <= 128 (sem Otsu);
- maior componente dentro da caixa, com os buracos preenchidos.
Cada máscara foi conferida visualmente na folha `images/reference/conferencia.png`.

Uso:
    python -m src.reference gerar      # gera as máscaras das fotos de teste com sufixo _01
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

from . import preprocessing, segmentation, split

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "..", "images", "reference")
INK_THRESHOLD = 128
SEARCH_PAD = 20


def reference_names(test_names):
    return sorted(n for n in test_names if n.endswith("_01.jpg"))


def reference_path(name):
    return os.path.join(REFERENCE_DIR, name.replace(".jpg", ".png"))


def _fill_holes(mask):
    labels, _ = segmentation.connected_components(~mask)
    border = np.setdiff1d(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])), [0])
    return ~np.isin(labels, border)


def build_reference(rgb):
    y0, y1, x0, x1 = segmentation.segment_corner(rgb)["naipe"]
    h, w = rgb.shape[:2]
    y0, x0 = max(0, y0 - SEARCH_PAD), max(0, x0 - SEARCH_PAD)
    y1, x1 = min(h, y1 + SEARCH_PAD + 1), min(w, x1 + SEARCH_PAD + 1)
    ink = rgb[y0:y1, x0:x1].min(axis=2) <= INK_THRESHOLD
    labels, components = segmentation.connected_components(ink)
    largest = max(components, key=lambda c: c["area"])["label"]
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = _fill_holes(labels == largest)
    return mask


def load_reference(name):
    return np.asarray(Image.open(reference_path(name))) > 0


def pipeline_mask(corner, shape):
    mask = np.zeros(shape, dtype=bool)
    oy, ox = corner["simbolo_origem"]
    m = corner["simbolo_mascara"] > 0
    mask[oy:oy + m.shape[0], ox:ox + m.shape[1]] = m
    return mask


def dice_iou(predicted, reference):
    inter = np.logical_and(predicted, reference).sum()
    total = predicted.sum() + reference.sum()
    union = np.logical_or(predicted, reference).sum()
    dice = 2 * inter / total if total else 1.0
    iou = inter / union if union else 1.0
    return float(dice), float(iou)


def contact_sheet(names, masks, path, size=160):
    cols = 5
    rows = (len(names) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, rows * (size + 16)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, name in enumerate(names):
        rgb = preprocessing.load_rgb(os.path.join(split.REAL_DIR, name))
        ys, xs = np.nonzero(masks[i])
        pad = 25
        y0, y1 = max(0, ys.min() - pad), ys.max() + pad
        x0, x1 = max(0, xs.min() - pad), xs.max() + pad
        crop = rgb[y0:y1, x0:x1].copy()
        m = masks[i][y0:y1, x0:x1]
        edge = m & ~segmentation.erode(m, 2)
        crop[edge] = (0, 200, 0)
        tile = Image.fromarray(crop)
        tile.thumbnail((size, size))
        r, c = divmod(i, cols)
        sheet.paste(tile, (c * size, r * (size + 16) + 16))
        draw.text((c * size + 3, r * (size + 16) + 2), name, fill="black")
    sheet.save(path)


def generate():
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    names = reference_names(split.load_split(split.REAL_SPLIT_PATH)["teste"])
    masks = []
    for name in names:
        mask = build_reference(preprocessing.load_rgb(os.path.join(split.REAL_DIR, name)))
        Image.fromarray(mask.astype(np.uint8) * 255).save(reference_path(name))
        masks.append(mask)
        print(f"{name}: {int(mask.sum())} px")
    contact_sheet(names, masks, os.path.join(REFERENCE_DIR, "conferencia.png"))


if __name__ == "__main__":
    if sys.argv[1:] == ["gerar"]:
        generate()
    else:
        print(__doc__)
