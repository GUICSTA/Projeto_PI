"""Experimentos da M3: robustez da segmentação (ablação) e condições de captura simuladas.

Experimento 4 — ablação do pré-processamento com ruído gaussiano.
    A configuração final é escolhida no conjunto de AJUSTE com σ = 60 e só depois medida no teste.
    Localização: IoU da caixa do naipe (referência = foto limpa) e Dice/IoU por pixel contra
    as máscaras de referência de `images/reference/`.
Experimento 5 — condições de captura simuladas sobre as 40 fotos de teste:
    fundo claro, iluminação fraca, gradiente de luz, rotação e escala reduzida.

Uso: python -m src.robustez   (gera images/results/experimentos_m3.json)
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image

from . import calibrate, classify, preprocessing, rank, reference, segmentation, split
from .experiments import IOU_MIN, NOISE_SEED, add_noise, box_iou

M3_PATH = os.path.join(calibrate.RESULTS_DIR, "experimentos_m3.json")
NOISE_LEVELS = (0, 30, 60)
SELECTION_SIGMA = 60
CONFIGS = {
    "baseline_m2": {"median": 0, "morphology": None},
    "abertura": {"median": 0, "morphology": "abertura"},
    "mediana_r1": {"median": 1, "morphology": None},
    "mediana_r2": {"median": 2, "morphology": None},
    "mediana_r1_abertura": {"median": 1, "morphology": "abertura"},
}
LIGHT_BACKGROUND = (205, 205, 200)


def _rate(hits, total):
    return round(hits / total, 4) if total else 0.0


def run(names, transform, config, rules, templates, references=None):
    """Roda o pipeline completo em cada imagem transformada e devolve as saídas por imagem."""
    outputs = {}
    for index, name in enumerate(names):
        rgb = transform(preprocessing.load_rgb(os.path.join(split.REAL_DIR, name)), index)
        start = time.perf_counter()
        try:
            extracted = classify.extract(rgb, morphology=config["morphology"], median=config["median"])
        except ValueError as error:
            outputs[name] = {"falha": str(error)}
            continue
        corner = extracted["canto"]
        if corner is None:
            outputs[name] = {"falha": "Carta não separada do fundo (imagem tratada como sem fundo)."}
            continue
        out = {
            "naipe": classify.decide(extracted["cor"], extracted["features"], rules),
            "valor": rank.recognize(corner, templates),
            "caixa": corner["naipe"],
            "tempo_ms": (time.perf_counter() - start) * 1000,
        }
        if references and name in references:
            out["dice_iou"] = reference.dice_iou(reference.pipeline_mask(corner, rgb.shape[:2]), references[name])
        outputs[name] = out
    return outputs


def summarize(names, outputs, clean_boxes=None, references=None):
    ok = [n for n in names if "falha" not in outputs[n]]
    summary = {
        "segmentadas": len(ok),
        "total": len(names),
        "acerto_naipe": _rate(sum(outputs[n]["naipe"] == split.suit_of(n) for n in ok), len(names)),
        "acerto_valor": _rate(sum(outputs[n]["valor"] == rank.rank_of(n) for n in ok), len(names)),
        "falhas": {n: outputs[n]["falha"] for n in names if n not in ok},
    }
    if clean_boxes:
        ious = [box_iou(outputs[n]["caixa"], clean_boxes[n]) if n in ok and clean_boxes.get(n) else 0.0 for n in names]
        summary["iou_medio_caixa"] = round(float(np.mean(ious)), 4)
        summary["localizado_iou_0_5"] = _rate(sum(i >= IOU_MIN for i in ious), len(names))
    if references:
        pairs = [outputs[n].get("dice_iou", (0.0, 0.0)) if n in ok else (0.0, 0.0) for n in references]
        summary["dice_medio_referencia"] = round(float(np.mean([p[0] for p in pairs])), 4)
        summary["iou_medio_referencia"] = round(float(np.mean([p[1] for p in pairs])), 4)
    times = [outputs[n]["tempo_ms"] for n in ok]
    summary["tempo_medio_ms"] = round(float(np.mean(times)), 1) if times else None
    return summary


def noise(sigma):
    return lambda rgb, index: add_noise(rgb, sigma, NOISE_SEED + index)


def light_background(rgb, index):
    card = segmentation.card_mask(rgb)
    out = rgb.copy()
    out[~card] = LIGHT_BACKGROUND
    return out


def dim(rgb, index):
    return np.clip(rgb.astype(np.float64) * 0.55, 0, 255).astype(np.uint8)


def light_gradient(rgb, index):
    h, w = rgb.shape[:2]
    ramp = np.linspace(1.0, 0.45, w)[None, :, None]
    return np.clip(rgb * ramp, 0, 255).astype(np.uint8)


def rotate(angle):
    def transform(rgb, index):
        fill = tuple(int(v) for v in np.median(rgb[:20, :20].reshape(-1, 3), axis=0))
        return np.asarray(Image.fromarray(rgb).rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=fill))
    return transform


def scale(factor):
    def transform(rgb, index):
        h, w = rgb.shape[:2]
        return np.asarray(Image.fromarray(rgb).resize((int(w * factor), int(h * factor)), Image.BILINEAR))
    return transform


CONDITIONS = {
    "original": lambda rgb, index: rgb,
    "fundo_claro": light_background,
    "iluminacao_fraca_55": dim,
    "gradiente_de_luz": light_gradient,
    "rotacao_10": rotate(10),
    "rotacao_25": rotate(25),
    "escala_50": scale(0.5),
}


def experiment_ablation(rules, templates, adjust_names, test_names, references):
    baseline = CONFIGS["baseline_m2"]
    clean_adjust = {n: o.get("caixa") for n, o in run(adjust_names, noise(0), baseline, rules, templates).items()}
    clean_test = {n: o.get("caixa") for n, o in run(test_names, noise(0), baseline, rules, templates).items()}

    selection = {
        name: summarize(adjust_names, run(adjust_names, noise(SELECTION_SIGMA), c, rules, templates), clean_adjust)
        for name, c in CONFIGS.items()
    }
    order = list(CONFIGS)
    chosen = max(order, key=lambda n: (selection[n]["localizado_iou_0_5"], selection[n]["iou_medio_caixa"], -order.index(n)))

    test = {}
    for sigma in NOISE_LEVELS:
        test[f"sigma_{sigma}"] = {
            name: summarize(test_names, run(test_names, noise(sigma), c, rules, templates, references), clean_test, references)
            for name, c in CONFIGS.items()
        }
    return {
        "configuracoes": CONFIGS,
        "selecao_sigma": SELECTION_SIGMA,
        "selecao_no_ajuste": selection,
        "configuracao_escolhida": chosen,
        "teste": test,
    }


def experiment_conditions(rules, templates, test_names, config_names):
    return {
        condition: {
            name: summarize(test_names, run(test_names, transform, CONFIGS[name], rules, templates))
            for name in config_names
        }
        for condition, transform in CONDITIONS.items()
    }


def main():
    division = split.load_split(split.REAL_SPLIT_PATH)
    rules = classify.load_rules()
    templates = rank.load_templates()
    names = reference.reference_names(division["teste"])
    references = {n: reference.load_reference(n) for n in names}

    ablation = experiment_ablation(rules, templates, division["ajuste"], division["teste"], references)
    chosen = ablation["configuracao_escolhida"]
    configs = ["baseline_m2"] + ([chosen] if chosen != "baseline_m2" else [])
    results = {
        "divisao": "split_reais.json",
        "parametros": {"ruido_semente": NOISE_SEED, "iou_minimo": IOU_MIN, "fundo_claro_rgb": LIGHT_BACKGROUND},
        "experimento_4_ablacao_ruido": ablation,
        "experimento_5_condicoes_simuladas": experiment_conditions(rules, templates, division["teste"], configs),
    }
    with open(M3_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"E4 escolhida no ajuste (sigma={SELECTION_SIGMA}): {chosen}")
    for name, r in ablation["selecao_no_ajuste"].items():
        print(f"  ajuste {name}: localizado {r['localizado_iou_0_5']:.0%}, IoU {r['iou_medio_caixa']:.2f}")
    for sigma, configs_result in ablation["teste"].items():
        for name, r in configs_result.items():
            print(
                f"E4 {sigma} {name}: seg {r['segmentadas']}/40, loc {r['localizado_iou_0_5']:.0%}, "
                f"Dice {r['dice_medio_referencia']:.3f}, naipe {r['acerto_naipe']:.0%}, "
                f"valor {r['acerto_valor']:.0%}, {r['tempo_medio_ms']} ms"
            )
    for condition, configs_result in results["experimento_5_condicoes_simuladas"].items():
        for name, r in configs_result.items():
            print(f"E5 {condition} {name}: seg {r['segmentadas']}/40, naipe {r['acerto_naipe']:.0%}, valor {r['acerto_valor']:.0%}")


if __name__ == "__main__":
    sys.exit(main())
