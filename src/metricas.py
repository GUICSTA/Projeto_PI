"""Precisão, revocação e F1 por classe a partir de uma matriz de confusão {esperado: {previsto: n}}."""


def per_class(matrix):
    classes = list(matrix)
    result = {}
    for c in classes:
        tp = matrix[c].get(c, 0)
        fp = sum(matrix[o].get(c, 0) for o in classes if o != c)
        fn = sum(n for p, n in matrix[c].items() if p != c)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if tp + fp + fn else None)
        result[c] = {
            "suporte": tp + fn,
            "precisao": None if precision is None else round(precision, 4),
            "revocacao": None if recall is None else round(recall, 4),
            "f1": None if f1 is None else round(f1, 4),
        }
    return result


def macro_f1(metrics):
    values = [m["f1"] for m in metrics.values() if m["suporte"] and m["f1"] is not None]
    return round(sum(values) / len(values), 4) if values else None
