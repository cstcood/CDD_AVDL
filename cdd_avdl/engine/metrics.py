"""Classification metrics reported in the paper (TABLE III).

The positive class is depression (label 1).  TPR and FPR are the two headline
numbers of the paper, so they are computed from the confusion matrix directly
rather than through a library.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Sequence

import torch

__all__ = ["Metrics", "compute_metrics", "confusion_matrix"]


@dataclass
class Metrics:
    """Accuracy, TPR, FPR, precision, recall and F1 for the positive class."""

    accuracy: float
    tpr: float
    fpr: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def confusion_matrix(
    predictions: Sequence[int],
    targets: Sequence[int],
    positive: int = 1,
) -> tuple:
    """Return (tp, fp, tn, fn) for the given positive label."""
    if len(predictions) != len(targets):
        raise ValueError(
            "predictions and targets differ in length: %d vs %d"
            % (len(predictions), len(targets))
        )
    tp = fp = tn = fn = 0
    for prediction, target in zip(predictions, targets):
        if target == positive:
            if prediction == positive:
                tp += 1
            else:
                fn += 1
        else:
            if prediction == positive:
                fp += 1
            else:
                tn += 1
    return tp, fp, tn, fn


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def compute_metrics(
    predictions: Sequence[int],
    targets: Sequence[int],
    positive: int = 1,
) -> Metrics:
    """Compute every reported metric from hard predictions."""
    tp, fp, tn, fn = confusion_matrix(predictions, targets, positive)
    total = tp + fp + tn + fn
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return Metrics(
        accuracy=_safe_divide(tp + tn, total),
        tpr=_safe_divide(tp, tp + fn),
        fpr=_safe_divide(fp, fp + tn),
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive: int = 1,
) -> Metrics:
    """Convenience wrapper that takes logits and integer targets."""
    predictions = logits.argmax(dim=1).detach().cpu().tolist()
    return compute_metrics(predictions, targets.detach().cpu().tolist(), positive)
