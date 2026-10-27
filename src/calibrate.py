import json
import os
import sys

from . import classify, preprocessing, split

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "images", "results")
EVALUATION_PATH = os.path.join(RESULTS_DIR, "avaliacao_reais.json")
COLOR_GROUPS = {"vermelho": classify.RED_SUITS, "preto": classify.BLACK_SUITS}


def extract_samples(filenames, directory=split.REAL_DIR):
    samples = []
    for name in filenames:
        extracted = classify.extract(preprocessing.load_rgb(os.path.join(directory, name)))
        samples.append({
            "arquivo": name,
            "naipe": split.suit_of(name),
            "cor": extracted["cor"],
            "features": extracted["features"],
        })
    return samples


def fit_rule(samples, suits):
    group = [s for s in samples if s["naipe"] in suits]
    if {s["naipe"] for s in group} != set(suits):
        raise ValueError(f"O ajuste precisa de amostras de {suits[0]} e {suits[1]}.")

    best = None
    for feature in sorted(group[0]["features"]):
        values = sorted({s["features"][feature] for s in group})
        spread = (values[-1] - values[0]) or 1.0
        for low, high in zip(values, values[1:]):
            limiar = (low + high) / 2
            margin = min(abs(s["features"][feature] - limiar) for s in group) / spread
            for above, below in (suits, suits[::-1]):
                hits = sum(
                    (above if s["features"][feature] > limiar else below) == s["naipe"]
                    for s in group
                )
                score = (hits / len(group), margin)
                if best is None or score > best[0]:
                    best = (score, {
                        "feature": feature,
                        "limiar": round(limiar, 4),
                        "acima": above,
                        "abaixo": below,
                        "acerto_ajuste": round(score[0], 4),
                    })
    return best[1]


def calibrate(samples):
    return {color: fit_rule(samples, suits) for color, suits in COLOR_GROUPS.items()}


def evaluate(rules, samples):
    predictions = []
    for s in samples:
        predicted = classify.decide(s["cor"], s["features"], rules)
        predictions.append({
            "arquivo": s["arquivo"],
            "naipe_esperado": s["naipe"],
            "naipe_previsto": predicted,
            "cor": s["cor"],
        })

    suits = sorted(set(classify.RED_SUITS + classify.BLACK_SUITS))
    confusion = {e: {p: 0 for p in suits} for e in suits}
    for p in predictions:
        confusion[p["naipe_esperado"]][p["naipe_previsto"]] += 1

    correct = sum(p["naipe_esperado"] == p["naipe_previsto"] for p in predictions)
    return {
        "acertos": correct,
        "total": len(predictions),
        "acuracia": round(correct / len(predictions), 4) if predictions else 0.0,
        "matriz_confusao": confusion,
        "previsoes": predictions,
    }


def _save(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    division = split.load_split(split.REAL_SPLIT_PATH)
    rules = calibrate(extract_samples(division["ajuste"]))
    _save({"divisao": "split_reais.json", "conjunto": "ajuste", "regras": rules},
          classify.CALIBRATED_RULES_PATH)

    result = evaluate(rules, extract_samples(division["teste"]))
    _save({"conjunto": "teste", **result}, EVALUATION_PATH)

    for color, rule in rules.items():
        print(f"{color}: {rule['feature']} > {rule['limiar']} -> {rule['acima']}, senão {rule['abaixo']}")
    print(f"Teste: {result['acertos']}/{result['total']} ({result['acuracia']:.0%})")


if __name__ == "__main__":
    sys.exit(main())
