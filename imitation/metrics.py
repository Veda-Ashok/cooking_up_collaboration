import numpy as np
from imitation.constants import ID_TO_ACTION


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num / den)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 6,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(y_true, y_pred):
        if 0 <= target < num_classes and 0 <= pred < num_classes:
            confusion[target, pred] += 1

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = _safe_div(correct, total)

    per_class = {}
    precision_values = []
    recall_values = []
    f1_values = []
    supports = []

    for class_id in range(num_classes):
        tp = float(confusion[class_id, class_id])
        fp = float(confusion[:, class_id].sum() - tp)
        fn = float(confusion[class_id, :].sum() - tp)
        support = int(confusion[class_id, :].sum())

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        name = ID_TO_ACTION.get(class_id, str(class_id))
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        supports.append(support)

    macro_precision = float(np.mean(precision_values))
    macro_recall = float(np.mean(recall_values))
    macro_f1 = float(np.mean(f1_values))

    support_arr = np.asarray(supports, dtype=np.float64)
    if support_arr.sum() == 0:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0
    else:
        weight = support_arr / support_arr.sum()
        weighted_precision = float(np.dot(weight, np.asarray(precision_values)))
        weighted_recall = float(np.dot(weight, np.asarray(recall_values)))
        weighted_f1 = float(np.dot(weight, np.asarray(f1_values)))

    return {
        "num_samples": int(y_true.shape[0]),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }

