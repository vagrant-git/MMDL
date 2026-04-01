from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .metrics import compute_classification_metrics


AGGREGATION_METHODS = ["majority_voting", "mean_probability_pooling", "logit_averaging"]


def _majority_vote(items: List[Dict[str, object]], num_classes: int) -> Tuple[int, List[float]]:
    votes = [int(item["pred_label"]) for item in items]
    counts = Counter(votes)
    probs = np.asarray([item["probabilities"] for item in items], dtype=np.float32)
    mean_probs = probs.mean(axis=0)
    best_label = max(range(num_classes), key=lambda cls: (counts.get(cls, 0), mean_probs[cls]))
    return int(best_label), mean_probs.tolist()


def _mean_probability_pooling(items: List[Dict[str, object]]) -> Tuple[int, List[float]]:
    probs = np.asarray([item["probabilities"] for item in items], dtype=np.float32)
    pooled = probs.mean(axis=0)
    return int(np.argmax(pooled)), pooled.tolist()


def _logit_averaging(items: List[Dict[str, object]]) -> Tuple[int, List[float]]:
    if all("logits" in item for item in items):
        logits = np.asarray([item["logits"] for item in items], dtype=np.float32)
        pooled = logits.mean(axis=0)
    else:
        probs = np.asarray([item["probabilities"] for item in items], dtype=np.float32)
        log_probs = np.log(np.clip(probs, 1e-8, 1.0))
        pooled = log_probs.mean(axis=0)
    return int(np.argmax(pooled)), pooled.tolist()


def aggregate_predictions_by_session(
    predictions: List[Dict[str, object]],
    method: str,
    num_classes: int = 5,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for item in predictions:
        grouped[str(item["session_id"])].append(item)

    session_predictions: List[Dict[str, object]] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for session_id, items in sorted(grouped.items()):
        true_label = int(items[0]["true_label"])
        if method == "majority_voting":
            pred_label, pooled_values = _majority_vote(items, num_classes=num_classes)
        elif method == "mean_probability_pooling":
            pred_label, pooled_values = _mean_probability_pooling(items)
        elif method == "logit_averaging":
            pred_label, pooled_values = _logit_averaging(items)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")
        session_predictions.append(
            {
                "session_id": session_id,
                "num_windows": len(items),
                "true_label": true_label,
                "pred_label": pred_label,
                "aggregation_method": method,
                "pooled_values": pooled_values,
            }
        )
        y_true.append(true_label)
        y_pred.append(pred_label)

    metrics = compute_classification_metrics(y_true, y_pred, labels=list(range(num_classes)))
    return session_predictions, metrics
