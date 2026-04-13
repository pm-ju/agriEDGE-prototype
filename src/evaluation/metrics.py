"""Agricultural detection metrics: TPR, FPR, and Advance Detection Time.

These metrics are SPECIFICALLY mentioned in the ESoC project description:
"implement transparent performance measures to ensure the quality of the
developed approaches, in terms of TPR, FPR, and advance detection time"

Advance Detection Time (ADT) is unique to this domain:
  How many milliseconds BEFORE the actual foreign object impact peak
  does the system trigger the alarm? Earlier = better = more time to
  stop the blades and prevent damage.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def compute_classification_metrics(y_true, y_pred, y_proba=None):
    """Compute standard classification metrics.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth labels (0 = normal, 1 = event).
    y_pred : np.ndarray of shape (n_samples,)
        Predicted labels.
    y_proba : np.ndarray of shape (n_samples,) or None
        Predicted probabilities for class 1 (for ROC/AUC).

    Returns
    -------
    metrics : dict
        Keys: "tpr", "fpr", "tnr", "fnr", "accuracy", "precision",
              "recall", "f1", "auc", "confusion_matrix"
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # sensitivity / recall
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "tnr": tn / (fp + tn) if (fp + tn) > 0 else 0.0,  # specificity
        "fnr": fn / (tp + fn) if (tp + fn) > 0 else 0.0,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Normal", "Foreign Object"]
        ),
    }

    if y_proba is not None:
        metrics["auc"] = roc_auc_score(y_true, y_proba)
        metrics["roc_curve"] = roc_curve(y_true, y_proba)
        metrics["pr_curve"] = precision_recall_curve(y_true, y_proba)

    return metrics


def compute_advance_detection_time(
    detected_events, true_event_peaks, sampling_rate_hz=1000
):
    """Compute Advance Detection Time (ADT).

    ADT measures how many milliseconds BEFORE the actual event peak
    the system first triggered an alarm. This is critical in agriculture:
    earlier detection → more time to engage the blade brake → less damage.

    Parameters
    ----------
    detected_events : list of dict
        Each dict has "start" (index), "end" (index), "confidence".
        Output of SlidingWindowDetector.detect().
    true_event_peaks : np.ndarray of int
        Ground truth indices where the foreign object impact peaks occur.
    sampling_rate_hz : int
        Sampling rate for converting indices to milliseconds.

    Returns
    -------
    adt_results : dict
        Keys: "mean_adt_ms", "median_adt_ms", "min_adt_ms", "max_adt_ms",
              "per_event_adt_ms" (list), "detection_rate" (fraction of events detected)
    """
    adts = []
    detected_count = 0

    for peak_idx in true_event_peaks:
        # Find the earliest detection that is before or overlaps the peak
        best_adt = None
        for det in detected_events:
            if det["start"] <= peak_idx:
                adt_samples = peak_idx - det["start"]
                adt_ms = adt_samples * 1000.0 / sampling_rate_hz
                if best_adt is None or adt_ms > best_adt:
                    best_adt = adt_ms
        if best_adt is not None:
            adts.append(best_adt)
            detected_count += 1

    if not adts:
        return {
            "mean_adt_ms": 0.0,
            "median_adt_ms": 0.0,
            "min_adt_ms": 0.0,
            "max_adt_ms": 0.0,
            "per_event_adt_ms": [],
            "detection_rate": 0.0,
        }

    return {
        "mean_adt_ms": float(np.mean(adts)),
        "median_adt_ms": float(np.median(adts)),
        "min_adt_ms": float(np.min(adts)),
        "max_adt_ms": float(np.max(adts)),
        "per_event_adt_ms": adts,
        "detection_rate": detected_count / len(true_event_peaks),
    }
