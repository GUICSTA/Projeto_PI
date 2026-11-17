import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import metricas


def test_per_class_precision_recall_f1():
    matrix = {"a": {"a": 3, "b": 1}, "b": {"a": 0, "b": 4}}
    result = metricas.per_class(matrix)
    assert result["a"] == {"suporte": 4, "precisao": 1.0, "revocacao": 0.75, "f1": 0.8571}
    assert result["b"]["precisao"] == 0.8
    assert metricas.macro_f1(result) == 0.873


def test_class_without_samples_is_ignored_in_macro_f1():
    matrix = {"a": {"a": 2, "b": 0}, "b": {"a": 0, "b": 0}}
    result = metricas.per_class(matrix)
    assert result["b"]["suporte"] == 0
    assert metricas.macro_f1(result) == 1.0
