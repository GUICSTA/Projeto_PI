import json
import os
import sys
import time

import numpy as np
from PIL import Image

from . import calibrate, classify, split

COMPARISON_PATH = os.path.join(calibrate.RESULTS_DIR, "comparacao_metodos.json")
TEMPLATE_SIZE = 32
KNN_K = 3
FEATURE_NAMES = ("fill_ratio", "vertical_symmetry", "width_profile_ratio")


def normalize_mask(mask, size=TEMPLATE_SIZE):
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    return np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.float64) / 255.0


def fit_rules(adjust):
    return calibrate.calibrate(adjust)


def predict_rules(model, sample):
    return classify.decide(sample["cor"], sample["features"], model)


def fit_templates(adjust):
    by_suit = {}
    for s in adjust:
        by_suit.setdefault(s["naipe"], []).append(normalize_mask(s["mascara"]))
    return {suit: np.mean(masks, axis=0) for suit, masks in by_suit.items()}


def predict_templates(model, sample):
    target = normalize_mask(sample["mascara"])
    candidates = [s for s in model if classify.color_of_suit(s) == sample["cor"]]
    return min(candidates, key=lambda suit: np.abs(model[suit] - target).mean())


def fit_knn(adjust):
    vectors = np.array([[s["features"][f] for f in FEATURE_NAMES] for s in adjust])
    mean, std = vectors.mean(axis=0), vectors.std(axis=0)
    std[std == 0] = 1.0
    return {
        "media": mean,
        "desvio": std,
        "vetores": (vectors - mean) / std,
        "naipes": [s["naipe"] for s in adjust],
    }


def predict_knn(model, sample, k=KNN_K):
    vector = (np.array([sample["features"][f] for f in FEATURE_NAMES]) - model["media"]) / model["desvio"]
    distances = np.linalg.norm(model["vetores"] - vector, axis=1)
    allowed = [i for i, suit in enumerate(model["naipes"]) if classify.color_of_suit(suit) == sample["cor"]]
    nearest = sorted(allowed, key=lambda i: distances[i])[:k]
    votes = {}
    for i in nearest:
        votes[model["naipes"][i]] = votes.get(model["naipes"][i], 0) + 1
    return max(votes, key=lambda suit: (votes[suit], -min(distances[i] for i in nearest if model["naipes"][i] == suit)))


METHODS = {
    "regras": (fit_rules, predict_rules),
    "template_matching": (fit_templates, predict_templates),
    "knn": (fit_knn, predict_knn),
}


def evaluate_method(predict, model, samples):
    suits = sorted(set(classify.RED_SUITS + classify.BLACK_SUITS))
    confusion = {e: {p: 0 for p in suits} for e in suits}
    errors = []
    start = time.perf_counter()
    for s in samples:
        predicted = predict(model, s)
        confusion[s["naipe"]][predicted] += 1
        if predicted != s["naipe"]:
            errors.append({"arquivo": s["arquivo"], "esperado": s["naipe"], "previsto": predicted})
    elapsed = time.perf_counter() - start

    correct = sum(confusion[suit][suit] for suit in suits)
    return {
        "acertos": correct,
        "total": len(samples),
        "acuracia": round(correct / len(samples), 4) if samples else 0.0,
        "matriz_confusao": confusion,
        "erros": errors,
        "tempo_decisao_ms": round(1000 * elapsed / max(1, len(samples)), 4),
    }


def compare(adjust, test):
    return {
        name: evaluate_method(predict, fit(adjust), test)
        for name, (fit, predict) in METHODS.items()
    }


def main():
    division = split.load_split(split.REAL_SPLIT_PATH)
    adjust = calibrate.extract_samples(division["ajuste"])
    test = calibrate.extract_samples(division["teste"])
    results = compare(adjust, test)

    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "divisao": "split_reais.json",
            "imagens_ajuste": len(adjust),
            "imagens_teste": len(test),
            "parametros": {"template_tamanho": TEMPLATE_SIZE, "knn_k": KNN_K, "knn_features": FEATURE_NAMES},
            "metodos": results,
        }, f, indent=2, ensure_ascii=False)
        f.write("\n")

    for name, result in results.items():
        print(f"{name}: {result['acertos']}/{result['total']} ({result['acuracia']:.0%})")


if __name__ == "__main__":
    sys.exit(main())
