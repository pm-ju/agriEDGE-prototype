"""Generate synthetic agricultural vibration data for the dashboard simulator.

This script creates realistic vibration time series that mimic what sensors
on a forage harvester would record:
- Normal operation: low-amplitude periodic vibration (blade rotation) + noise
- Foreign object event: a sharp transient spike superimposed on normal vibration

The synthetic data is used ONLY for the real-time dashboard demo.
The actual ML training uses the FordA dataset.

Usage:
    python data/synthetic/generate_synthetic.py

Output:
    data/synthetic/normal_samples.npy    — 100 normal vibration samples
    data/synthetic/event_samples.npy     — 100 samples with foreign object events
    data/synthetic/event_timestamps.npy  — The exact timestamp (in ms) of each event
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    SYNTHETIC_DATA_DIR,
    SAMPLING_RATE_HZ,
    SIGNAL_DURATION_SEC,
    NOISE_AMPLITUDE,
    EVENT_AMPLITUDE_RANGE,
    EVENT_DURATION_MS,
    RANDOM_SEED,
)


def generate_normal_vibration(n_samples=100):
    """Generate normal harvester vibration signals.

    Model: sum of blade rotation harmonics + Gaussian noise.
    f_blade ≈ 20 Hz (typical chopper drum speed), with harmonics at 40, 60 Hz.

    Parameters
    ----------
    n_samples : int
        Number of normal samples to generate.

    Returns
    -------
    signals : np.ndarray of shape (n_samples, n_timesteps)
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_timesteps = SAMPLING_RATE_HZ * SIGNAL_DURATION_SEC
    t = np.linspace(0, SIGNAL_DURATION_SEC, n_timesteps)

    signals = []
    for _ in range(n_samples):
        # Blade rotation harmonics with slight random phase/amplitude variation
        f_base = 20 + rng.normal(0, 0.5)  # ~20 Hz ± jitter
        signal = (
            0.15 * np.sin(2 * np.pi * f_base * t + rng.uniform(0, 2 * np.pi))
            + 0.08 * np.sin(2 * np.pi * 2 * f_base * t + rng.uniform(0, 2 * np.pi))
            + 0.04 * np.sin(2 * np.pi * 3 * f_base * t + rng.uniform(0, 2 * np.pi))
        )
        # Add Gaussian noise (general machine vibration)
        signal += rng.normal(0, NOISE_AMPLITUDE, n_timesteps)
        signals.append(signal)

    return np.array(signals, dtype=np.float32)


def generate_event_vibration(n_samples=100):
    """Generate vibration signals containing a foreign object impact event.

    The event is modeled as a damped impulse (exponentially decaying sinusoid)
    superimposed on normal vibration at a random time position.

    Parameters
    ----------
    n_samples : int
        Number of event samples to generate.

    Returns
    -------
    signals : np.ndarray of shape (n_samples, n_timesteps)
    event_positions : np.ndarray of shape (n_samples,)
        The index (timestamp) where each event starts.
    """
    rng = np.random.default_rng(RANDOM_SEED + 1)
    n_timesteps = SAMPLING_RATE_HZ * SIGNAL_DURATION_SEC

    # Start with normal vibration
    normal = generate_normal_vibration(n_samples)
    signals = normal.copy()
    event_positions = []

    for i in range(n_samples):
        # Random event position (not too close to edges)
        margin = int(0.1 * n_timesteps)
        event_start = rng.integers(margin, n_timesteps - margin)
        event_positions.append(event_start)

        # Event duration in samples
        duration_ms = rng.integers(*EVENT_DURATION_MS)
        duration_samples = int(duration_ms * SAMPLING_RATE_HZ / 1000)

        # Damped impulse: A * sin(2πf*t) * exp(-decay*t)
        amplitude = rng.uniform(*EVENT_AMPLITUDE_RANGE)
        f_impact = rng.uniform(150, 400)  # High-frequency impact
        decay = rng.uniform(30, 80)
        t_event = np.arange(duration_samples) / SAMPLING_RATE_HZ
        impulse = amplitude * np.sin(2 * np.pi * f_impact * t_event) * np.exp(
            -decay * t_event
        )

        # Superimpose on normal vibration
        end_idx = min(event_start + duration_samples, n_timesteps)
        actual_len = end_idx - event_start
        signals[i, event_start:end_idx] += impulse[:actual_len]

    return signals, np.array(event_positions)


def main():
    """Generate and save all synthetic data."""
    print("Generating normal vibration samples...")
    normal = generate_normal_vibration(100)
    np.save(SYNTHETIC_DATA_DIR / "normal_samples.npy", normal)
    print(f"  Saved: {normal.shape}")

    print("Generating foreign object event samples...")
    events, positions = generate_event_vibration(100)
    np.save(SYNTHETIC_DATA_DIR / "event_samples.npy", events)
    np.save(SYNTHETIC_DATA_DIR / "event_timestamps.npy", positions)
    print(f"  Saved: {events.shape}, events at {positions[:5]}...")

    print("Done! Synthetic data saved to:", SYNTHETIC_DATA_DIR)


if __name__ == "__main__":
    main()
