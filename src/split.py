import json
import os
import random
import sys

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
INPUT_DIR = os.path.join(IMAGES_DIR, "input", "sinteticas")
SPLIT_PATH = os.path.join(IMAGES_DIR, "split.json")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DEFAULT_TEST_RATIO = 0.4
DEFAULT_SEED = 42


def suit_of(filename):
    return os.path.basename(filename).split("_")[0]


def list_images(directory=INPUT_DIR):
    return sorted(
        f for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def create_split(filenames, test_ratio=DEFAULT_TEST_RATIO, seed=DEFAULT_SEED):
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio deve estar entre 0 e 1.")

    by_suit = {}
    for name in sorted(filenames):
        by_suit.setdefault(suit_of(name), []).append(name)

    rng = random.Random(seed)
    adjust, test = [], []
    for suit in sorted(by_suit):
        items = list(by_suit[suit])
        if len(items) < 2:
            raise ValueError(f"Naipe '{suit}' precisa de ao menos 2 imagens para dividir.")
        rng.shuffle(items)
        n_test = min(len(items) - 1, max(1, round(len(items) * test_ratio)))
        test.extend(items[:n_test])
        adjust.extend(items[n_test:])

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


def main():
    split = create_split(list_images())
    save_split(split)
    print(f"Ajuste: {len(split['ajuste'])} imagens | Teste: {len(split['teste'])} imagens")
    print(f"Divisão salva em {os.path.abspath(SPLIT_PATH)}")


if __name__ == "__main__":
    sys.exit(main())
