import json
import os
import random
import sys

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
INPUT_DIR = os.path.join(IMAGES_DIR, "input", "sinteticas")
SPLIT_PATH = os.path.join(IMAGES_DIR, "split.json")
REAL_DIR = os.path.join(IMAGES_DIR, "input", "reais")
REAL_SPLIT_PATH = os.path.join(IMAGES_DIR, "split_reais.json")
UNSUITED = {"coringa"}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DEFAULT_TEST_RATIO = 0.4
DEFAULT_SEED = 42


def suit_of(filename):
    return os.path.basename(filename).split("_")[0]


def card_of(filename):
    parts = os.path.splitext(os.path.basename(filename))[0].split("_")
    return "_".join(parts[:2]) if len(parts) >= 3 else "_".join(parts)


def list_images(directory=INPUT_DIR):
    return sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTENSIONS) and suit_of(f) not in UNSUITED
    )


def create_split(filenames, test_ratio=DEFAULT_TEST_RATIO, seed=DEFAULT_SEED):
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio deve estar entre 0 e 1.")

    by_suit = {}
    for name in sorted(filenames):
        by_suit.setdefault(suit_of(name), {}).setdefault(card_of(name), []).append(name)

    rng = random.Random(seed)
    adjust, test = [], []
    for suit in sorted(by_suit):
        cards = sorted(by_suit[suit])
        if len(cards) < 2:
            raise ValueError(f"Naipe '{suit}' precisa de ao menos 2 cartas para dividir.")
        rng.shuffle(cards)
        n_test = min(len(cards) - 1, max(1, round(len(cards) * test_ratio)))
        for card in cards[:n_test]:
            test.extend(by_suit[suit][card])
        for card in cards[n_test:]:
            adjust.extend(by_suit[suit][card])

    return {
        "seed": seed,
        "test_ratio": test_ratio,
        "ajuste": sorted(adjust),
        "teste": sorted(test),
    }


def save_split(split, path=SPLIT_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_split(path=SPLIT_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    directory, path = (REAL_DIR, REAL_SPLIT_PATH) if argv[:1] == ["reais"] else (INPUT_DIR, SPLIT_PATH)
    split = create_split(list_images(directory))
    save_split(split, path)
    print(f"Ajuste: {len(split['ajuste'])} imagens | Teste: {len(split['teste'])} imagens")
    print(f"Divisão salva em {os.path.abspath(path)}")


if __name__ == "__main__":
    sys.exit(main())
