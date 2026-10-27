import csv
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

from . import classify, experiments, features, preprocessing, segmentation, split

INTERMEDIATE_DIR = os.path.join(os.path.dirname(__file__), "..", "images", "results", "intermediarias")
FEATURES_TABLE_PATH = os.path.join(os.path.dirname(__file__), "..", "images", "results", "tabela_features.csv")
EXAMPLE_CASES = {
    "copas_A_01.jpg": "sucesso com sombra sobre a carta",
    "espadas_10_01.jpg": "valor com múltiplos elementos",
    "paus_K_01.jpg": "figura com desenho próximo ao índice",
    "paus_J_01.jpg": "carta inclinada",
    "copas_10_01.jpg": "carta de cabeça para baixo",
    "coringa_preto_01.jpg": "falha esperada: coringa sem naipe",
}
NOISY_CASE = ("ouros_5_01.jpg", 60)
PALETTE = np.array([
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40),
], dtype=np.uint8)


def colorize_labels(labels):
    image = np.full(labels.shape + (3,), 255, dtype=np.uint8)
    for label in np.unique(labels):
        if label:
            image[labels == label] = PALETTE[(label - 1) % len(PALETTE)]
    return image


def _draw_box(draw, bbox, origin, color, width):
    y0, y1, x0, x1 = bbox
    draw.rectangle([x0 - origin[1], y0 - origin[0], x1 - origin[1], y1 - origin[0]], outline=color, width=width)


def save_intermediate(rgb, out_dir, morphology=None):
    os.makedirs(out_dir, exist_ok=True)
    card = segmentation.card_mask(rgb)
    Image.fromarray(card.astype(np.uint8) * 255).resize((card.shape[1] // 4, card.shape[0] // 4)).save(
        os.path.join(out_dir, "1_mascara_carta.png"))

    rgb_crop, inside, region = segmentation.corner_region(rgb, card)
    Image.fromarray(rgb_crop).save(os.path.join(out_dir, "2_canto.png"))

    labels, elements, _ = segmentation.corner_elements(rgb_crop, inside, morphology)
    Image.fromarray(colorize_labels(labels)).save(os.path.join(out_dir, "3_componentes_rotulados.png"))

    boxes = Image.fromarray(rgb_crop).copy()
    draw = ImageDraw.Draw(boxes)
    for element in elements:
        _draw_box(draw, element["bbox"], (0, 0), (255, 190, 0), 2)
    try:
        corner = segmentation.segment_corner(rgb, card, morphology=morphology)
    except ValueError as error:
        boxes.save(os.path.join(out_dir, "4_valor_e_naipe.png"))
        return {"erro": str(error)}

    origin = (region[0], region[2])
    _draw_box(draw, corner["valor"], origin, (0, 170, 0), 3)
    _draw_box(draw, corner["naipe"], origin, (0, 0, 255), 3)
    boxes.save(os.path.join(out_dir, "4_valor_e_naipe.png"))
    Image.fromarray(corner["simbolo_mascara"]).save(os.path.join(out_dir, "5_mascara_simbolo.png"))
    return corner


def features_table(directory=split.REAL_DIR, path=FEATURES_TABLE_PATH):
    rules = classify.load_rules()
    rows = []
    for name in sorted(os.listdir(directory)):
        row = {"arquivo": name, "naipe_esperado": split.suit_of(name)}
        try:
            corner = segmentation.segment_corner(preprocessing.load_rgb(os.path.join(directory, name)))
        except ValueError as error:
            row["status"] = f"falha: {error}"
            rows.append(row)
            continue
        feats = features.extract_features(corner["simbolo_mascara"])
        color = segmentation.dominant_color(corner["simbolo_rgb"], corner["simbolo_mascara"])
        y0, y1, x0, x1 = corner["naipe"]
        row.update({
            "status": "ok",
            "cor": color,
            "naipe_previsto": classify.decide(color, feats, rules),
            "simbolo_altura_px": y1 - y0 + 1,
            "simbolo_largura_px": x1 - x0 + 1,
            "simbolo_area_px": int(np.count_nonzero(corner["simbolo_mascara"])),
            "elementos_no_canto": len(corner["elementos"]),
            "threshold_otsu": int(corner["threshold"]),
            **{k: round(v, 4) for k, v in feats.items()},
        })
        rows.append(row)

    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    for name, description in EXAMPLE_CASES.items():
        rgb = preprocessing.load_rgb(os.path.join(split.REAL_DIR, name))
        result = save_intermediate(rgb, os.path.join(INTERMEDIATE_DIR, os.path.splitext(name)[0]))
        print(f"{name} ({description}): {'falha - ' + result['erro'] if 'erro' in result else 'ok'}")

    name, sigma = NOISY_CASE
    noisy = experiments.add_noise(preprocessing.load_rgb(os.path.join(split.REAL_DIR, name)), sigma, experiments.NOISE_SEED)
    for morphology in (None, "abertura"):
        folder = f"{os.path.splitext(name)[0]}_ruido{sigma}_{morphology or 'sem_morfologia'}"
        result = save_intermediate(noisy, os.path.join(INTERMEDIATE_DIR, folder), morphology)
        print(f"{folder}: {'falha - ' + result['erro'] if 'erro' in result else 'ok'}")

    rows = features_table()
    print(f"Tabela de features: {len(rows)} imagens em {os.path.abspath(FEATURES_TABLE_PATH)}")


if __name__ == "__main__":
    sys.exit(main())
