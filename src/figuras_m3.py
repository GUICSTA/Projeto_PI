"""Saídas intermediárias dos casos de sucesso e falha da M3 e análise de erros da rotação de 25°.

Uso: python -m src.figuras_m3
Gera images/results/intermediarias_m3/<caso>/ e images/results/erros_rotacao_25.json.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

from . import classify, preprocessing, rank, robustez, split, visualize

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "images", "results", "intermediarias_m3")
ERRORS_PATH = os.path.join(os.path.dirname(__file__), "..", "images", "results", "erros_rotacao_25.json")
CASES = {
    "ouros_4_01_iluminacao_fraca_final": ("ouros_4_01.jpg", robustez.dim, classify.FINAL_CONFIG),
    "ouros_4_01_iluminacao_fraca_baseline_m2": ("ouros_4_01.jpg", robustez.dim, classify.BASELINE_M2_CONFIG),
    "ouros_5_01_ruido60_final": ("ouros_5_01.jpg", robustez.noise(60), classify.FINAL_CONFIG),
    "copas_8_01_fundo_claro_final": ("copas_8_01.jpg", robustez.light_background, classify.FINAL_CONFIG),
    "espadas_7_01_rotacao_25_final": ("espadas_7_01.jpg", robustez.rotate(25), classify.FINAL_CONFIG),
    "paus_3_01_rotacao_25_final": ("paus_3_01.jpg", robustez.rotate(25), classify.FINAL_CONFIG),
}


def rotation_errors(names, rules, templates):
    errors = []
    for index, name in enumerate(names):
        rgb = robustez.rotate(25)(preprocessing.load_rgb(os.path.join(split.REAL_DIR, name)), index)
        try:
            extracted = classify.extract(rgb, **classify.FINAL_CONFIG)
        except ValueError as error:
            errors.append({"arquivo": name, "falha": str(error)})
            continue
        suit = classify.decide(extracted["cor"], extracted["features"], rules)
        value = rank.recognize(extracted["canto"], templates)
        if suit != split.suit_of(name) or value != rank.rank_of(name):
            errors.append({
                "arquivo": name,
                "naipe": [split.suit_of(name), suit],
                "valor": [rank.rank_of(name), value],
                "width_profile_ratio": round(float(extracted["features"]["width_profile_ratio"]), 3),
            })
    return errors


def main():
    for case, (name, transform, config) in CASES.items():
        rgb = transform(preprocessing.load_rgb(os.path.join(split.REAL_DIR, name)), 0)
        out = os.path.join(OUT_DIR, case)
        os.makedirs(out, exist_ok=True)
        Image.fromarray(rgb).resize((rgb.shape[1] // 4, rgb.shape[0] // 4)).save(os.path.join(out, "0_entrada.png"))
        result = visualize.save_intermediate(rgb, out, config)
        print(f"{case}: {result.get('erro', 'ok') if isinstance(result, dict) else 'ok'}")

    division = split.load_split(split.REAL_SPLIT_PATH)
    errors = rotation_errors(division["teste"], classify.load_rules(), rank.load_templates())
    with open(ERRORS_PATH, "w", encoding="utf-8") as f:
        json.dump({"configuracao": "final", "rotacao_graus": 25, "erros": errors}, f, indent=2, ensure_ascii=False)
        f.write("\n")
    for e in errors:
        print(e)


if __name__ == "__main__":
    sys.exit(main())
