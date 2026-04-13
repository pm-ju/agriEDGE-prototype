"""Sliding window event detector.

Converts a per-window classifier into a continuous-time event detector.
This is the core of the "real-time detection" capability:

1. Take a continuous sensor stream
2. Extract overlapping windows of fixed size
3. Classify each window using the trained sktime model
4. Apply a detection threshold to get binary alerts
5. Merge adjacent detections into event regions
6. Calculate "advance detection time" — how early before the peak we detected

This simulates exactly what would run on the embedded agricultural ECU.
"""

import numpy as np


class SlidingWindowDetector:
    """Sliding window event detector wrapping an sktime classifier.

    Parameters
    ----------
    classifier : sktime classifier (fitted)
        Must have a .predict_proba(X) method.
    window_size : int
        Number of time steps per window.
    stride : int
        Step size between consecutive windows.
    threshold : float
        Probability threshold above which we trigger a detection.
    """

    def __init__(self, classifier, window_size=50, stride=10, threshold=0.5):
        self.classifier = classifier
        self.window_size = window_size
        self.stride = stride
        self.threshold = threshold

    def detect(self, signal):
        """Run detection on a 1D signal.

        Parameters
        ----------
        signal : np.ndarray of shape (n_timesteps,)
            Raw sensor signal (single channel).

        Returns
        -------
        detections : list of dict
            Each detection has keys:
            - "start": start index in the signal
            - "end": end index in the signal
            - "confidence": max probability in the detection region
            - "timestamp_ms": start time in milliseconds (assuming 1kHz sampling)
        probabilities : np.ndarray of shape (n_windows,)
            Per-window detection probability.
        window_centers : np.ndarray of shape (n_windows,)
            Center index of each window in the original signal.
        """
        n = len(signal)
        windows = []
        centers = []

        for start in range(0, n - self.window_size + 1, self.stride):
            end = start + self.window_size
            windows.append(signal[start:end])
            centers.append(start + self.window_size // 2)

        if not windows:
            return [], np.array([]), np.array([])

        # Shape: (n_windows, 1, window_size) — sktime 3D format
        X = np.array(windows)[:, np.newaxis, :]

        # Get probability of class 1 (foreign object)
        proba = self.classifier.predict_proba(X)[:, 1]

        # Threshold to binary detections
        detected = proba >= self.threshold

        # Merge adjacent detections into event regions
        detections = self._merge_detections(detected, proba, centers)

        return detections, proba, np.array(centers)

    def _merge_detections(self, detected, proba, centers):
        """Merge adjacent detection windows into contiguous event regions.

        Parameters
        ----------
        detected : np.ndarray of bool
        proba : np.ndarray of float
        centers : list of int

        Returns
        -------
        events : list of dict
        """
        events = []
        i = 0
        while i < len(detected):
            if detected[i]:
                start_idx = i
                max_conf = proba[i]
                while i < len(detected) and detected[i]:
                    max_conf = max(max_conf, proba[i])
                    i += 1
                end_idx = i - 1
                events.append(
                    {
                        "start": centers[start_idx] - self.window_size // 2,
                        "end": centers[end_idx] + self.window_size // 2,
                        "confidence": float(max_conf),
                        "timestamp_ms": centers[start_idx],
                    }
                )
            else:
                i += 1
        return events
