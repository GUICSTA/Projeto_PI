import json
import os

from . import preprocessing, segmentation, features

RED_SUITS = ("copas", "ouros")
BLACK_SUITS = ("espadas", "paus")
CALIBRATED_RULES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "images", "results", "limiares_reais.json"
)
SYNTHETIC_RULES = {
    "vermelho": {"feature": "vertical_symmetry", "limiar": 0.75, "acima": "ouros", "abaixo": "copas"},
    "preto": {"feature": "width_profile_ratio", "limiar": 0.9, "acima": "espadas", "abaixo": "paus"},
}


def color_of_suit(suit):
    return "vermelho" if suit in RED_SUITS else "preto"


def load_rules(path=CALIBRATED_RULES_PATH):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)["regras"]


def extract(rgb):
    card = segmentation.card_mask(rgb)
    photo = segmentation.has_background(card)

    if photo:
        corner = segmentation.segment_corner(rgb, card)
        rgb_crop, mask_crop = corner["simbolo_rgb"], corner["simbolo_mascara"]
        threshold = corner["threshold"]
    else:
        gray = preprocessing.to_grayscale(rgb)
        binary, threshold = preprocessing.binarize(gray)
        rgb_crop, mask_crop = segmentation.crop_symbol(rgb, binary)

    return {
        "foto": photo,
        "cor": segmentation.dominant_color(rgb_crop, mask_crop),
        "threshold_otsu": threshold,
        "features": features.extract_features(mask_crop),
        "mascara": mask_crop,
    }


def decide(color, feats, rules):
    rule = rules[color]
    return rule["acima"] if feats[rule["feature"]] > rule["limiar"] else rule["abaixo"]


def classify_suit(image_path, rules=None):
    extracted = extract(preprocessing.load_rgb(image_path))
    if rules is None:
        rules = (load_rules() if extracted["foto"] else None) or SYNTHETIC_RULES

    return {
        "naipe": decide(extracted["cor"], extracted["features"], rules),
        "cor": extracted["cor"],
        "threshold_otsu": extracted["threshold_otsu"],
        "features": extracted["features"],
    }
