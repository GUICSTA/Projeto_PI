import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import classify, compare


def _mask(kind):
    mask = np.zeros((40, 40), dtype=np.uint8)
    if kind == "topo":
        mask[5:20, 10:30] = 255
    else:
        mask[20:35, 10:30] = 255
    return mask


def _sample(suit, kind, ratio):
    return {
        "arquivo": f"{suit}.jpg",
        "naipe": suit,
        "cor": classify.color_of_suit(suit),
        "features": {"fill_ratio": 0.5, "vertical_symmetry": 0.8, "width_profile_ratio": ratio},
        "mascara": _mask(kind),
    }


ADJUST = [
    _sample("espadas", "topo", 0.70), _sample("espadas", "topo", 0.72),
    _sample("paus", "base", 0.90), _sample("paus", "base", 0.92),
    _sample("copas", "topo", 2.2), _sample("copas", "topo", 2.3),
    _sample("ouros", "base", 1.0), _sample("ouros", "base", 1.05),
]
TEST = [_sample("espadas", "topo", 0.71), _sample("paus", "base", 0.91), _sample("ouros", "base", 1.02)]


def test_template_matching_picks_closest_template_of_same_color():
    model = compare.fit_templates(ADJUST)
    assert compare.predict_templates(model, TEST[0]) == "espadas"
    assert compare.predict_templates(model, TEST[2]) == "ouros"


def test_knn_votes_among_same_color():
    model = compare.fit_knn(ADJUST)
    assert compare.predict_knn(model, TEST[1]) == "paus"
    assert compare.predict_knn(model, TEST[2]) == "ouros"


def test_compare_evaluates_all_methods_on_same_images():
    results = compare.compare(ADJUST, TEST)
    assert set(results) == {"regras", "template_matching", "knn"}
    for result in results.values():
        assert result["total"] == len(TEST)
        assert sum(sum(row.values()) for row in result["matriz_confusao"].values()) == len(TEST)


def test_saved_comparison_uses_same_test_set():
    with open(compare.COMPARISON_PATH, encoding="utf-8") as f:
        saved = json.load(f)
    totals = {r["total"] for r in saved["metodos"].values()}
    assert totals == {saved["imagens_teste"]}
