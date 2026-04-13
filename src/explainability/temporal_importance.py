"""Temporal importance analysis for detection explainability.

Answers the question: "Which part of the sensor signal was most important
for triggering the foreign object detection?"

This directly addresses the ESoC stretch goal:
"apply explainable AI or root cause analysis techniques to the pipeline,
e.g., detecting which sensors or which part of the event was relevant
for the detection"

Method: Occlusion-based temporal importance.
For each time window, replace it with zeros (or mean) and measure how much
the detection probability drops. Higher drop = more important region.
"""

import numpy as np


def compute_temporal_importance(
    classifier, signal, window_size=20, stride=5, baseline="mean"
):
    """Compute temporal importance scores via occlusion sensitivity.

    Parameters
    ----------
    classifier : sktime classifier (fitted)
        Must have .predict_proba(X) method.
    signal : np.ndarray of shape (n_timesteps,)
        The 1D sensor signal to explain.
    window_size : int
        Size of the occlusion window.
    stride : int
        Step between occlusion positions.
    baseline : str, "mean" or "zero"
        What to replace the occluded region with.

    Returns
    -------
    importance_scores : np.ndarray of shape (n_positions,)
        Importance score for each position. Higher = more important for detection.
    positions : np.ndarray of shape (n_positions,)
        Center position of each occlusion window.
    """
    # Get the original prediction probability
    X_orig = signal[np.newaxis, np.newaxis, :]  # (1, 1, n_timesteps)
    original_proba = classifier.predict_proba(X_orig)[0, 1]

    fill_value = signal.mean() if baseline == "mean" else 0.0

    scores = []
    positions = []

    for start in range(0, len(signal) - window_size + 1, stride):
        end = start + window_size
        # Create occluded version
        occluded = signal.copy()
        occluded[start:end] = fill_value

        X_occ = occluded[np.newaxis, np.newaxis, :]
        occluded_proba = classifier.predict_proba(X_occ)[0, 1]

        # Importance = drop in detection probability when this region is removed
        importance = original_proba - occluded_proba
        scores.append(importance)
        positions.append(start + window_size // 2)

    return np.array(scores), np.array(positions)
