import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import experiments


def test_add_noise_is_reproducible_and_bounded():
    rgb = np.full((10, 10, 3), 128, dtype=np.uint8)
    first = experiments.add_noise(rgb, 50, 3)
    assert np.array_equal(first, experiments.add_noise(rgb, 50, 3))
    assert first.dtype == np.uint8 and not np.array_equal(first, rgb)


def test_zero_noise_keeps_image():
    rgb = np.full((4, 4, 3), 200, dtype=np.uint8)
    assert experiments.add_noise(rgb, 0, 1) is rgb


def test_saved_experiments_use_same_test_set():
    with open(experiments.EXPERIMENTS_PATH, encoding="utf-8") as f:
        saved = json.load(f)
    total = saved["imagens_teste"]
    assert all(r["total"] == total for r in saved["experimento_1_limiares"].values())
    for r in saved["experimento_2_morfologia_com_ruido"].values():
        assert r["segmentacao_ok"] + len(r["falhas_segmentacao"]) == total


def test_box_iou():
    assert experiments.box_iou((0, 9, 0, 9), (0, 9, 0, 9)) == 1.0
    assert experiments.box_iou((0, 9, 0, 9), (20, 29, 20, 29)) == 0.0
    assert abs(experiments.box_iou((0, 9, 0, 9), (0, 9, 5, 14)) - 50 / 150) < 1e-9
