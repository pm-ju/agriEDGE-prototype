"""Load FordA and map it to the project input format."""

import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from sktime.datasets import load_UCR_UEA_dataset

from src.config import DATASET_NAME, RAW_DATA_DIR


def load_ford_a():
    """Load and preprocess the FordA dataset.

    Returns
    -------
    X_train : np.ndarray of shape (n_train, 1, 500)
        Training time series in sktime 3D format (n_instances, n_dims, series_length).
    y_train : np.ndarray of shape (n_train,)
        Training labels, 0 = normal, 1 = foreign object event.
    X_test : np.ndarray of shape (n_test, 1, 500)
        Test time series.
    y_test : np.ndarray of shape (n_test,)
        Test labels.
    """
    dataset_cache_dir = RAW_DATA_DIR / DATASET_NAME
    temp_dir = RAW_DATA_DIR / "_tmp"
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    with _project_tempdir(temp_dir):
        X_train, y_train = load_UCR_UEA_dataset(
            DATASET_NAME,
            split="train",
            return_type="numpy3d",
            extract_path=dataset_cache_dir,
        )
        X_test, y_test = load_UCR_UEA_dataset(
            DATASET_NAME,
            split="test",
            return_type="numpy3d",
            extract_path=dataset_cache_dir,
        )

    # Convert labels: {-1, 1} → {0, 1}
    y_train = np.where(y_train.astype(int) == -1, 0, 1)
    y_test = np.where(y_test.astype(int) == -1, 0, 1)

    # Normalize each instance independently (z-score)
    X_train = _normalize(X_train)
    X_test = _normalize(X_test)

    return X_train.astype(np.float32), y_train, X_test.astype(np.float32), y_test


def _normalize(X):
    """Apply per-instance z-score normalization.

    Parameters
    ----------
    X : np.ndarray of shape (n_instances, n_dims, series_length)

    Returns
    -------
    X_norm : np.ndarray, same shape, zero mean and unit variance per instance.
    """
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True)
    std[std == 0] = 1.0  # prevent division by zero for constant signals
    return (X - mean) / std


@contextmanager
def _project_tempdir(temp_dir):
    """Redirect tempfile usage to a writable project-local directory."""
    previous_tempdir = tempfile.tempdir
    previous_mkdtemp = tempfile.mkdtemp
    previous_env = {name: os.environ.get(name) for name in ("TMP", "TEMP", "TMPDIR")}
    temp_dir_str = str(temp_dir)

    tempfile.tempdir = temp_dir_str
    tempfile.mkdtemp = lambda suffix=None, prefix=None, dir=None: _safe_mkdtemp(
        dir=dir or temp_dir_str,
        prefix=prefix or "tmp",
        suffix=suffix or "",
    )
    for name in previous_env:
        os.environ[name] = temp_dir_str

    try:
        yield
    finally:
        tempfile.tempdir = previous_tempdir
        tempfile.mkdtemp = previous_mkdtemp
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _safe_mkdtemp(dir, prefix, suffix):
    """Create a temp directory without relying on Windows mkdtemp ACL behavior."""
    base_dir = RAW_DATA_DIR / "_tmp" if dir is None else Path(dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    while True:
        candidate = base_dir / f"{prefix}{uuid.uuid4().hex}{suffix}"
        try:
            candidate.mkdir()
            return str(candidate)
        except FileExistsError:
            continue
