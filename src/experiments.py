import json
import os
import sys

import numpy as np

from . import calibrate, classify, compare, preprocessing, segmentation, split

EXPERIMENTS_PATH = os.path.join(calibrate.RESULTS_DIR, "experimentos.json")
NOISE_LEVELS = (0, 30, 60)
NOISE_SEED = 7
MORPHOLOGY_CONFIGS = (None, "abertura")
IOU_MIN = 0.5


def add_noise(rgb, sigma, seed):
    if sigma == 0:
        return rgb
    rng = np.random.default_rng(seed)
    noisy = rgb.astype(np.float64) + rng.normal(0, sigma, rgb.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def box_iou(a, b):
    y0, y1 = max(a[0], b[0]), min(a[1], b[1])
    x0, x1 = max(a[2], b[2]), min(a[3], b[3])
    if y1 < y0 or x1 < x0:
        return 0.0
    inter = (y1 - y0 + 1) * (x1 - x0 + 1)
    area = lambda box: (box[1] - box[0] + 1) * (box[3] - box[2] + 1)
    return inter / (area(a) + area(b) - inter)


def symbol_boxes(filenames, sigma, morphology=None, directory=split.REAL_DIR):
    boxes = {}
    for index, name in enumerate(filenames):
        rgb = add_noise(preprocessing.load_rgb(os.path.join(directory, name)), sigma, NOISE_SEED + index)
        try:
            boxes[name] = segmentation.segment_corner(rgb, morphology=morphology)["naipe"]
        except ValueError:
            boxes[name] = None
    return boxes


def noisy_samples(filenames, sigma, morphology=None, directory=split.REAL_DIR):
    samples, failures = [], []
    for index, name in enumerate(filenames):
        rgb = add_noise(preprocessing.load_rgb(os.path.join(directory, name)), sigma, NOISE_SEED + index)
        try:
            extracted = classify.extract(rgb, morphology=morphology)
        except ValueError as error:
            failures.append({"arquivo": name, "motivo": str(error)})
            continue
        samples.append({
            "arquivo": name,
            "naipe": split.suit_of(name),
            "cor": extracted["cor"],
            "features": extracted["features"],
            "mascara": extracted["mascara"],
        })
    return samples, failures


def _accuracy(hits, total):
    return round(hits / total, 4) if total else 0.0


def experiment_thresholds(rules, test):
    result = {}
    for name, current in (("limiares_sinteticos", classify.SYNTHETIC_RULES), ("limiares_calibrados", rules)):
        evaluation = calibrate.evaluate(current, test)
        result[name] = {
            "acertos": evaluation["acertos"],
            "total": evaluation["total"],
            "acuracia": evaluation["acuracia"],
            "matriz_confusao": evaluation["matriz_confusao"],
        }
    return result


def experiment_morphology(rules, test_names):
    reference = symbol_boxes(test_names, 0)
    result = {}
    for sigma in NOISE_LEVELS:
        for morphology in MORPHOLOGY_CONFIGS:
            boxes = symbol_boxes(test_names, sigma, morphology)
            ious = [box_iou(boxes[n], reference[n]) if boxes[n] else 0.0 for n in test_names]
            samples, failures = noisy_samples(test_names, sigma, morphology)
            color_hits = sum(s["cor"] == classify.color_of_suit(s["naipe"]) for s in samples)
            suit_hits = sum(classify.decide(s["cor"], s["features"], rules) == s["naipe"] for s in samples)
            result[f"sigma_{sigma}_{morphology or 'sem_morfologia'}"] = {
                "sigma": sigma,
                "morfologia": morphology or "nenhuma",
                "segmentacao_ok": len(samples),
                "falhas_segmentacao": failures,
                "proporcao_segmentada": _accuracy(len(samples), len(test_names)),
                "iou_medio_naipe": round(float(np.mean(ious)), 4),
                "naipe_localizado_iou_0_5": _accuracy(sum(i >= IOU_MIN for i in ious), len(test_names)),
                "acerto_cor": _accuracy(color_hits, len(test_names)),
                "acerto_naipe": _accuracy(suit_hits, len(test_names)),
            }
    return result


def experiment_methods_with_noise(adjust, test_names):
    models = {name: fit(adjust) for name, (fit, _) in compare.METHODS.items()}
    result = {}
    for sigma in NOISE_LEVELS:
        samples, failures = noisy_samples(test_names, sigma)
        result[f"sigma_{sigma}"] = {
            name: {
                "acertos": sum(predict(models[name], s) == s["naipe"] for s in samples),
                "total": len(test_names),
                "acuracia": _accuracy(sum(predict(models[name], s) == s["naipe"] for s in samples), len(test_names)),
            }
            for name, (_, predict) in compare.METHODS.items()
        }
        result[f"sigma_{sigma}"]["falhas_segmentacao"] = len(failures)
    return result


def main():
    division = split.load_split(split.REAL_SPLIT_PATH)
    adjust = calibrate.extract_samples(division["ajuste"])
    test = calibrate.extract_samples(division["teste"])
    rules = calibrate.calibrate(adjust)

    results = {
        "divisao": "split_reais.json",
        "imagens_ajuste": len(adjust),
        "imagens_teste": len(test),
        "parametros": {"ruido_sigma": NOISE_LEVELS, "ruido_semente": NOISE_SEED, "raio_morfologia": 1, "iou_minimo": IOU_MIN},
        "experimento_1_limiares": experiment_thresholds(rules, test),
        "experimento_2_morfologia_com_ruido": experiment_morphology(rules, division["teste"]),
        "experimento_3_metodos_com_ruido": experiment_methods_with_noise(adjust, division["teste"]),
    }
    with open(EXPERIMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")

    for name, r in results["experimento_1_limiares"].items():
        print(f"E1 {name}: {r['acertos']}/{r['total']}")
    for name, r in results["experimento_2_morfologia_com_ruido"].items():
        print(f"E2 {name}: segmentadas {r['segmentacao_ok']}, IoU {r['iou_medio_naipe']:.2f}, localizado {r['naipe_localizado_iou_0_5']:.0%}, cor {r['acerto_cor']:.0%}, naipe {r['acerto_naipe']:.0%}")
    for name, r in results["experimento_3_metodos_com_ruido"].items():
        print(f"E3 {name}: " + ", ".join(f"{m} {v['acuracia']:.0%}" for m, v in r.items() if isinstance(v, dict)))


if __name__ == "__main__":
    sys.exit(main())
