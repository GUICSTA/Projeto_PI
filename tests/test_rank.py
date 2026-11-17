import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import preprocessing, rank, segmentation, split

REAL_DIR = os.path.join(os.path.dirname(__file__), "..", "images", "input", "reais")


def _corner(name):
    return segmentation.segment_corner(preprocessing.load_rgb(os.path.join(REAL_DIR, name)))


def test_rank_of_reads_filename():
    assert rank.rank_of("copas_10_01.jpg") == "10"
    assert rank.rank_of("paus_Q_02.jpg") == "Q"


def test_hole_areas_counts_closed_regions():
    mask = np.zeros((40, 20), dtype=np.uint8)
    mask[:, :] = 255
    mask[5:15, 5:15] = 0
    mask[25:35, 5:15] = 0
    assert len(rank.hole_areas(mask)) == 2
    assert rank.hole_areas(np.full((10, 10), 255, dtype=np.uint8)) == []


def test_value_mask_of_ten_has_two_elements():
    assert _corner("espadas_10_01.jpg")["valor_elementos"] == 2


def test_structural_rules_take_precedence():
    template = np.zeros((rank.TEMPLATE_SIZE, rank.TEMPLATE_SIZE))
    templates = {"6": template, "Q": template}
    ten = {"elementos": 2, "buracos": [100], "template": template}
    eight = {"elementos": 1, "buracos": [100, 80], "template": template}
    queen = {"elementos": 1, "buracos": [100, 20], "template": template}
    assert rank.predict(templates, ten) == "10"
    assert rank.predict(templates, eight) == "8"
    assert rank.predict(templates, queen) in templates


def test_rank_recognition_on_test_split():
    division = split.load_split(split.REAL_SPLIT_PATH)
    adjust, _ = rank.extract_values(division["ajuste"])
    test, failures = rank.extract_values(division["teste"])
    result = rank.evaluate(rank.fit(adjust), test, len(division["teste"]))
    assert not failures
    assert result["acuracia"] >= 0.9
