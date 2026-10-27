import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import split

SUITS = ["copas", "ouros", "paus", "espadas"]
FILENAMES = [f"{suit}_{i:02d}.png" for suit in SUITS for i in range(5)]


def test_split_has_no_overlap_and_covers_all_images():
    result = split.create_split(FILENAMES)
    adjust, test = set(result["ajuste"]), set(result["teste"])
    assert adjust.isdisjoint(test)
    assert adjust | test == set(FILENAMES)


def test_split_is_stratified_by_suit():
    result = split.create_split(FILENAMES)
    for suit in SUITS:
        assert any(split.suit_of(f) == suit for f in result["ajuste"])
        assert any(split.suit_of(f) == suit for f in result["teste"])


def test_split_is_deterministic_for_same_seed():
    assert split.create_split(FILENAMES, seed=7) == split.create_split(FILENAMES, seed=7)


def test_split_changes_with_different_seed():
    first = split.create_split(FILENAMES, seed=1)
    second = split.create_split(FILENAMES, seed=2)
    assert first["teste"] != second["teste"]


def test_split_rejects_invalid_ratio():
    with pytest.raises(ValueError):
        split.create_split(FILENAMES, test_ratio=1.0)


def test_split_rejects_suit_with_single_image():
    with pytest.raises(ValueError):
        split.create_split(["copas_00.png"])


def test_saved_split_matches_images_on_disk():
    saved = split.load_split()
    on_disk = set(split.list_images())
    assert set(saved["ajuste"]).isdisjoint(saved["teste"])
    assert set(saved["ajuste"]) | set(saved["teste"]) == on_disk
