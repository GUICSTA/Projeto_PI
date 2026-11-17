"""Reconhecimento do valor da carta (A, 2–10, J, Q, K) a partir do valor já localizado no canto.

Decisão:
1. regras estruturais: 2 elementos → "10"; 2 buracos de tamanho parecido (razão ≥ 0,5) → "8".
   O Q também tem 2 buracos (a perna fecha um buraco pequeno, razão ≈ 0,21 no ajuste),
   por isso a razão entre as áreas é exigida;
2. template matching: a máscara do valor é redimensionada para 32×32 e comparada com a média
   das máscaras de cada valor no conjunto de ajuste (menor diferença absoluta média).

Uso: python -m src.rank   (gera images/results/avaliacao_valor.json)
"""
import json
import os
import sys

import numpy as np
from PIL import Image

from . import metricas, preprocessing, segmentation, split

RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
TEMPLATE_SIZE = 32
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "images", "results")
RANK_PATH = os.path.join(RESULTS_DIR, "avaliacao_valor.json")
TEMPLATES_PATH = os.path.join(RESULTS_DIR, "templates_valor.npz")


def rank_of(filename):
    return os.path.basename(filename).split("_")[1]


def normalize(mask):
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    return np.asarray(img.resize((TEMPLATE_SIZE, TEMPLATE_SIZE), Image.BILINEAR), dtype=np.float64) / 255


EIGHT_HOLE_RATIO = 0.5


def hole_areas(mask):
    padded = np.pad(mask > 0, 1)
    labels, components = segmentation.connected_components(~padded)
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    min_area = 0.005 * padded.size
    return sorted((c["area"] for c in components if c["label"] not in border and c["area"] >= min_area), reverse=True)


def describe(corner):
    return {
        "elementos": corner["valor_elementos"],
        "buracos": hole_areas(corner["valor_mascara"]),
        "template": normalize(corner["valor_mascara"]),
    }


def extract_values(filenames, directory=split.REAL_DIR, median=0):
    samples, failures = [], []
    for name in filenames:
        try:
            corner = segmentation.segment_corner(preprocessing.load_rgb(os.path.join(directory, name)), median=median)
        except ValueError as error:
            failures.append({"arquivo": name, "motivo": str(error)})
            continue
        samples.append({"arquivo": name, "valor": rank_of(name), **describe(corner)})
    return samples, failures


def fit(samples):
    templates = {}
    for rank in RANKS:
        masks = [s["template"] for s in samples if s["valor"] == rank]
        if masks:
            templates[rank] = np.mean(masks, axis=0)
    return templates


def save_templates(templates, path=TEMPLATES_PATH):
    np.savez_compressed(path, **templates)


def load_templates(path=TEMPLATES_PATH):
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        return {r: data[r] for r in RANKS if r in data}


def recognize(corner, templates):
    return predict(templates, describe(corner))


def predict(templates, sample, structural=True):
    if structural:
        if sample["elementos"] == 2:
            return "10"
        holes = sample["buracos"]
        if len(holes) == 2 and holes[1] / holes[0] >= EIGHT_HOLE_RATIO:
            return "8"
    candidates = {r: t for r, t in templates.items() if not structural or r not in ("10", "8")}
    return min(candidates, key=lambda r: np.abs(candidates[r] - sample["template"]).mean())


def evaluate(templates, samples, total, structural=True):
    matrix = {e: {p: 0 for p in RANKS} for e in RANKS}
    hits = 0
    errors = []
    for s in samples:
        predicted = predict(templates, s, structural)
        matrix[s["valor"]][predicted] += 1
        if predicted == s["valor"]:
            hits += 1
        else:
            errors.append({"arquivo": s["arquivo"], "esperado": s["valor"], "previsto": predicted})
    return {
        "acertos": hits,
        "total": total,
        "acuracia": round(hits / total, 4) if total else 0.0,
        "erros": errors,
        "metricas_por_classe": metricas.per_class(matrix),
        "f1_macro": metricas.macro_f1(metricas.per_class(matrix)),
        "matriz_confusao": matrix,
    }


def main():
    division = split.load_split(split.REAL_SPLIT_PATH)
    adjust, _ = extract_values(division["ajuste"])
    test, failures = extract_values(division["teste"])
    templates = fit(adjust)
    save_templates(templates)

    results = {
        "divisao": "split_reais.json",
        "valores_com_template_no_ajuste": sorted(templates, key=RANKS.index),
        "parametros": {"tamanho_template": TEMPLATE_SIZE, "regras_estruturais": {"2 elementos": "10", "2 buracos com razao de area >= 0.5": "8"}},
        "falhas_segmentacao_teste": failures,
        "baseline_so_template": evaluate(templates, test, len(division["teste"]), structural=False),
        "template_com_regras_estruturais": evaluate(templates, test, len(division["teste"]), structural=True),
    }
    with open(RANK_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")

    for name in ("baseline_so_template", "template_com_regras_estruturais"):
        r = results[name]
        print(f"{name}: {r['acertos']}/{r['total']} ({r['acuracia']:.1%})")
        for e in r["erros"]:
            print(f"  {e['arquivo']}: esperado {e['esperado']}, previsto {e['previsto']}")


if __name__ == "__main__":
    sys.exit(main())
