"""Sliding-window detector built around a per-window classifier."""

import numpy as np
from sktime.detection.base import BaseDetector


class SlidingWindowDetector(BaseDetector):
    """Sliding window event detector extending sktime BaseDetector.

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

    _tags = {
        "task": "anomaly_detection",
        "learning_type": "supervised",
        "capability:missing_values": False,
        "capability:multivariate": False,
    }

    def __init__(self, classifier, window_size=50, stride=10, threshold=0.5):
        self.classifier = classifier
        self.window_size = window_size
        self.stride = stride
        self.threshold = threshold
        
        # State buffer for continuous stream mode
        self._stream_buffer = []
        self._stream_global_time = 0
        super().__init__()

    def _fit(self, X, y=None):
        """Fit the detection sequence (no-op as underlying classifier is pre-fitted)."""
        return self

    def _predict(self, X):
        """Offline batch detection conforming to sktime signature."""
        signal = X.values.flatten() if hasattr(X, "values") else np.array(X).flatten()
        detections, _, _ = self.detect(signal)
        return detections

    def updateStream(self, new_chunk):
        """True continuous-time stream processing mode.
        
        Ingests asynchronous signal chunks (e.g. from a live queue),
        evaluates completed windows, and yields overlapping detections.
        """
        self._stream_buffer.extend(new_chunk)
        stream_events = []
        
        while len(self._stream_buffer) >= self.window_size:
            window = np.array(self._stream_buffer[:self.window_size])[np.newaxis, np.newaxis, :]
            
            # In an actual embedded setup, this runs the compiled ONNX session
            proba = self.classifier.predict_proba(window)[0, 1]
            
            if proba >= self.threshold:
                center = self._stream_global_time + self.window_size // 2
                stream_events.append({
                    "start": center - self.window_size // 2,
                    "end": center + self.window_size // 2,
                    "confidence": float(proba),
                    "timestamp_ms": center,
                })
            
            self._stream_buffer = self._stream_buffer[self.stride:]
            self._stream_global_time += self.stride
            
        return self._merge_detections_stream(stream_events)

    def _merge_detections_stream(self, events):
        """Merges adjacent raw stream detections."""
        if not events:
            return []
        merged = [events[0].copy()]
        for e in events[1:]:
            last = merged[-1]
            # If windows overlap or are directly adjacent based on stride
            if e["start"] <= last["end"] + self.stride:
                last["end"] = e["end"]
                last["confidence"] = max(last["confidence"], e["confidence"])
            else:
                merged.append(e.copy())
        return merged

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
