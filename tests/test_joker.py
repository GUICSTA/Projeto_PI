import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import classify, preprocessing, segmentation

REAL_DIR = os.path.join(os.path.dirname(__file__), "..", "images", "input", "reais")
JOKERS = sorted(f for f in os.listdir(REAL_DIR) if f.startswith("coringa"))


@pytest.mark.parametrize("name", JOKERS)
def test_joker_is_detected(name):
    rgb = preprocessing.load_rgb(os.path.join(REAL_DIR, name))
    with pytest.raises(segmentation.JokerFound):
        segmentation.segment_corner(rgb)
    assert classify.classify_suit(os.path.join(REAL_DIR, name))["naipe"] == "coringa"


@pytest.mark.parametrize("name", ["copas_Q_01.jpg", "espadas_10_01.jpg", "paus_K_01.jpg"])
def test_regular_card_is_not_joker(name):
    assert classify.classify_suit(os.path.join(REAL_DIR, name))["naipe"] != "coringa"
