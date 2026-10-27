"""
Testes do pipeline de identificação de naipe.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import generate_synthetic_cards, classify, preprocessing, segmentation, features

SYNTH_DIR = os.path.join(
    os.path.dirname(__file__), "..", "images", "input", "sinteticas"
)


@pytest.fixture(scope="module", autouse=True)
def ensure_synthetic_images():
    if not os.path.isdir(SYNTH_DIR) or len(os.listdir(SYNTH_DIR)) == 0:
        generate_synthetic_cards.generate()


def _paths_for(suit):
    return [
        os.path.join(SYNTH_DIR, f)
        for f in os.listdir(SYNTH_DIR)
        if f.startswith(suit)
    ]


def test_generates_images_for_all_suits():
    for suit in ["copas", "ouros", "paus", "espadas"]:
        assert len(_paths_for(suit)) > 0


def test_preprocessing_binarizes_symbol():
    path = _paths_for("copas")[0]
    rgb = preprocessing.load_rgb(path)
    gray = preprocessing.to_grayscale(rgb)
    binary, threshold = preprocessing.binarize(gray)
    assert binary.shape == gray.shape
    assert 0 <= threshold <= 255
    assert binary.max() == 255  # existe algum pixel de símbolo


def test_segmentation_crops_symbol_region():
    path = _paths_for("espadas")[0]
    rgb = preprocessing.load_rgb(path)
    gray = preprocessing.to_grayscale(rgb)
    binary, _ = preprocessing.binarize(gray)
    rgb_crop, mask_crop = segmentation.crop_symbol(rgb, binary)
    assert rgb_crop.shape[0] < rgb.shape[0]
    assert mask_crop.max() == 255


def test_color_detection_red_suits():
    for suit in ["copas", "ouros"]:
        path = _paths_for(suit)[0]
        rgb = preprocessing.load_rgb(path)
        gray = preprocessing.to_grayscale(rgb)
        binary, _ = preprocessing.binarize(gray)
        rgb_crop, mask_crop = segmentation.crop_symbol(rgb, binary)
        color = segmentation.dominant_color(rgb_crop, mask_crop)
        assert color == "vermelho"


def test_color_detection_black_suits():
    for suit in ["paus", "espadas"]:
        path = _paths_for(suit)[0]
        rgb = preprocessing.load_rgb(path)
        gray = preprocessing.to_grayscale(rgb)
        binary, _ = preprocessing.binarize(gray)
        rgb_crop, mask_crop = segmentation.crop_symbol(rgb, binary)
        color = segmentation.dominant_color(rgb_crop, mask_crop)
        assert color == "preto"


@pytest.mark.parametrize("suit", ["copas", "ouros", "paus", "espadas"])
def test_full_pipeline_classifies_suit(suit):
    """
    Testa o pipeline completo em TODAS as amostras sintéticas de cada naipe
    (não só a primeira), para dar um sinal mais realista de acerto.
    """
    paths = _paths_for(suit)
    correct = 0
    for path in paths:
        result = classify.classify_suit(path)
        if result["naipe"] == suit:
            correct += 1

    accuracy = correct / len(paths)
    # threshold observado durante o desenvolvimento com imagens sintéticas
    assert accuracy >= 0.6, (
        f"Acurácia baixa para {suit}: {accuracy:.2f} "
        f"(esperado nas sintéticas; fotos reais vão exigir recalibração)"
    )


def test_main_runs_on_real_photo(tmp_path, capsys):
    from src import main
    photo = os.path.join(os.path.dirname(__file__), "..", "images", "input", "reais", "copas_Q_01.jpg")
    main.main([photo, "--saida", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Naipe identificado: copas" in out
    assert (tmp_path / "4_valor_e_naipe.png").exists()
