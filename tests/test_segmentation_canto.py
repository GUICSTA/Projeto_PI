import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import preprocessing, segmentation

REAL_DIR = os.path.join(os.path.dirname(__file__), "..", "images", "input", "reais")
SYNTH_DIR = os.path.join(os.path.dirname(__file__), "..", "images", "input", "sinteticas")
RED_SUITS = {"copas", "ouros"}
SUIT_PHOTOS = sorted(f for f in os.listdir(REAL_DIR) if not f.startswith("coringa"))
JOKER_PHOTOS = sorted(f for f in os.listdir(REAL_DIR) if f.startswith("coringa"))


def _load(name, directory=REAL_DIR):
    return preprocessing.load_rgb(os.path.join(directory, name))


def _inside(inner, outer):
    return (
        inner[0] >= outer[0] and inner[1] <= outer[1]
        and inner[2] >= outer[2] and inner[3] <= outer[3]
    )


def test_connected_components_separates_regions():
    binary = np.zeros((10, 10), dtype=np.uint8)
    binary[1:3, 1:3] = 255
    binary[6:9, 5:9] = 255
    binary[3, 3] = 255
    _, components = segmentation.connected_components(binary)
    areas = sorted(c["area"] for c in components)
    assert areas == [5, 12]


def test_card_mask_excludes_background():
    rgb = _load(SUIT_PHOTOS[0])
    card = segmentation.card_mask(rgb)
    assert card.shape == rgb.shape[:2]
    assert segmentation.has_background(card)
    assert 0.2 < card.mean() < 0.9
    assert not card[0, 0]


def test_synthetic_image_has_no_background():
    rgb = _load(sorted(os.listdir(SYNTH_DIR))[0], SYNTH_DIR)
    assert not segmentation.has_background(segmentation.card_mask(rgb))


@pytest.mark.parametrize("name", SUIT_PHOTOS)
def test_corner_symbol_is_below_value_with_suit_color(name):
    rgb = _load(name)
    corner = segmentation.segment_corner(rgb)
    value, symbol = corner["valor"], corner["naipe"]

    assert _inside(value, corner["regiao"])
    assert _inside(symbol, corner["regiao"])
    assert symbol[0] > value[1]
    assert min(symbol[3], value[3]) >= max(symbol[2], value[2])
    assert corner["simbolo_mascara"].max() == 255

    color = segmentation.dominant_color(corner["simbolo_rgb"], corner["simbolo_mascara"])
    expected = "vermelho" if name.split("_")[0] in RED_SUITS else "preto"
    assert color == expected


def test_value_groups_multiple_elements_and_ignores_others():
    one = {"label": 1, "area": 300, "bbox": (10, 60, 10, 20)}
    zero = {"label": 2, "area": 900, "bbox": (12, 60, 26, 50)}
    symbol = {"label": 3, "area": 1200, "bbox": (75, 115, 12, 48)}
    side = {"label": 4, "area": 500, "bbox": (80, 110, 70, 90)}
    lower = {"label": 5, "area": 400, "bbox": (130, 150, 15, 40)}

    value, found = segmentation.split_value_and_symbol([lower, side, symbol, zero, one])
    assert sorted(c["label"] for c in value) == [1, 2]
    assert found["label"] == 3


def test_split_without_symbol_raises():
    with pytest.raises(ValueError):
        segmentation.split_value_and_symbol([{"label": 1, "area": 300, "bbox": (10, 60, 10, 20)}])


@pytest.mark.parametrize("name", JOKER_PHOTOS)
def test_joker_has_no_suit_symbol(name):
    with pytest.raises(ValueError):
        segmentation.segment_corner(_load(name))


def test_opening_removes_isolated_noise_and_keeps_object():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    mask[1, 1] = True
    opened = segmentation.opening(mask, 1)
    assert not opened[1, 1]
    assert opened[6:14, 5:15].all() and opened[5:15, 6:14].all()


def test_closing_fills_small_gap():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    mask[10, 10] = False
    assert segmentation.closing(mask, 1)[10, 10]


@pytest.mark.parametrize("morphology", ["abertura", "fechamento"])
def test_corner_segmentation_with_morphology(morphology):
    corner = segmentation.segment_corner(_load(SUIT_PHOTOS[0]), morphology=morphology)
    assert corner["naipe"][0] > corner["valor"][1]
