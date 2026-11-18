"""Experimento 8 — regras × template matching × k-NN nas condições de captura simuladas.

Responde ao feedback da M2: repetir a comparação dos três métodos num teste diferente do original,
com matriz de confusão por naipe, macro-F1, tempo relativo e erros de segmentação.

- Os três modelos são ajustados exatamente como na M2 (`src/compare.py`): mesmas 64 fotos de ajuste,
  mesmos parâmetros (template 32×32, k = 3) e mesma semente da divisão (42).
- As 40 fotos de teste passam por cada condição de `src/robustez.py` e são segmentadas com a
  configuração final (`classify.FINAL_CONFIG`). A mesma segmentação alimenta os três métodos.
- Falha de segmentação conta como erro, numa coluna própria da matriz ("falha").

Uso: python -m src.metodos_condicoes   (gera images/results/metodos_condicoes.json)
"""
import json
import os
import sys
import time

from . import calibrate, classify, compare, metricas, preprocessing, robustez, split

OUTPUT_PATH = os.path.join(calibrate.RESULTS_DIR, "metodos_condicoes.json")
SUITS = sorted(set(classify.RED_SUITS + classify.BLACK_SUITS))
FAILURE = "falha"


def condition_samples(names, transform):
    samples, failures = [], []
    for index, name in enumerate(names):
        rgb = transform(preprocessing.load_rgb(os.path.join(split.REAL_DIR, name)), index)
        try:
            extracted = classify.extract(rgb, **classify.FINAL_CONFIG)
            if extracted["canto"] is None:
                raise ValueError("Carta não separada do fundo (imagem tratada como sem fundo).")
        except ValueError as error:
            failures.append({"arquivo": name, "naipe": split.suit_of(name), "motivo": str(error)})
            continue
        samples.append({
            "arquivo": name,
            "naipe": split.suit_of(name),
            "cor": extracted["cor"],
            "features": extracted["features"],
            "mascara": extracted["mascara"],
        })
    return samples, failures


def evaluate(predict, model, samples, failures):
    matrix = {e: {p: 0 for p in SUITS + [FAILURE]} for e in SUITS}
    errors = []
    start = time.perf_counter()
    for s in samples:
        predicted = predict(model, s)
        matrix[s["naipe"]][predicted] += 1
        if predicted != s["naipe"]:
            errors.append({"arquivo": s["arquivo"], "esperado": s["naipe"], "previsto": predicted})
    elapsed = time.perf_counter() - start
    for f in failures:
        matrix[f["naipe"]][FAILURE] += 1

    total = len(samples) + len(failures)
    hits = sum(matrix[s][s] for s in SUITS)
    per_class = metricas.per_class(matrix)
    return {
        "acertos": hits,
        "total": total,
        "acuracia": round(hits / total, 4) if total else 0.0,
        "f1_macro": metricas.macro_f1(per_class),
        "metricas_por_classe": per_class,
        "matriz_confusao": matrix,
        "erros_decisao": errors,
        "tempo_decisao_ms": round(1000 * elapsed / max(1, len(samples)), 4),
    }


def main():
    division = split.load_split(split.REAL_SPLIT_PATH)
    adjust = calibrate.extract_samples(division["ajuste"])
    models = {name: fit(adjust) for name, (fit, _) in compare.METHODS.items()}

    results = {}
    conditions = dict(robustez.CONDITIONS)
    conditions["ruido_30"] = robustez.noise(30)
    conditions["ruido_60"] = robustez.noise(60)
    for condition, transform in conditions.items():
        samples, failures = condition_samples(division["teste"], transform)
        results[condition] = {
            "falhas_segmentacao": failures,
            "metodos": {
                name: evaluate(predict, models[name], samples, failures)
                for name, (_, predict) in compare.METHODS.items()
            },
        }
        line = ", ".join(
            f"{m} {r['acuracia']:.1%} (F1 {r['f1_macro']})" for m, r in results[condition]["metodos"].items()
        )
        print(f"{condition}: falhas {len(failures)} | {line}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "divisao": "split_reais.json",
            "ajuste": "64 fotos, extração da M2 (compare.py), parâmetros da M2",
            "teste": "40 fotos de teste em cada condição, segmentação final (classify.FINAL_CONFIG)",
            "parametros": {"template_tamanho": compare.TEMPLATE_SIZE, "knn_k": compare.KNN_K,
                           "ruido_semente": robustez.NOISE_SEED},
            "condicoes": results,
        }, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    sys.exit(main())
