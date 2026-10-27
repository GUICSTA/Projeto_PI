import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import calibrate, classify, split


def _sample(suit, ratio, symmetry):
    return {
        "arquivo": f"{suit}.jpg",
        "naipe": suit,
        "cor": classify.color_of_suit(suit),
        "features": {"fill_ratio": 0.5, "vertical_symmetry": symmetry, "width_profile_ratio": ratio},
    }


def test_fit_rule_finds_feature_and_direction():
    samples = [_sample("espadas", r, 0.78) for r in (0.68, 0.70, 0.72)]
    samples += [_sample("paus", r, 0.78) for r in (0.89, 0.91, 0.93)]
    rule = calibrate.fit_rule(samples, classify.BLACK_SUITS)
    assert rule["feature"] == "width_profile_ratio"
    assert rule["acima"] == "paus" and rule["abaixo"] == "espadas"
    assert 0.72 < rule["limiar"] < 0.89
    assert rule["acerto_ajuste"] == 1.0


def test_evaluate_counts_hits_and_confusion():
    rules = {
        "vermelho": {"feature": "vertical_symmetry", "limiar": 0.8, "acima": "ouros", "abaixo": "copas"},
        "preto": {"feature": "width_profile_ratio", "limiar": 0.8, "acima": "paus", "abaixo": "espadas"},
    }
    samples = [_sample("ouros", 1.0, 0.95), _sample("copas", 2.2, 0.95), _sample("paus", 0.9, 0.77)]
    result = calibrate.evaluate(rules, samples)
    assert result["acertos"] == 2 and result["total"] == 3
    assert result["matriz_confusao"]["copas"]["ouros"] == 1


def test_calibrated_rules_generalize_to_test_set():
    rules = classify.load_rules()
    assert rules is not None
    division = split.load_split(split.REAL_SPLIT_PATH)
    result = calibrate.evaluate(rules, calibrate.extract_samples(division["teste"]))
    assert result["acuracia"] >= 0.9
