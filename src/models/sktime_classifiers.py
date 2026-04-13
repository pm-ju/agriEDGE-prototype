"""sktime classifier wrappers for foreign object detection."""

from collections.abc import Callable, Sequence

import numpy as np
import torch.nn as nn
from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.classification.interval_based import TimeSeriesForestClassifier

from src.config import (
    MLP_BATCH_SIZE,
    MLP_CRITERION,
    MLP_DROPOUT,
    MLP_HIDDEN_DIM,
    MLP_LR,
    MLP_N_LAYERS,
    MLP_NUM_EPOCHS,
    MLP_OPTIMIZER,
    RANDOM_SEED,
)


class ProjectMLPNetworkTorch(nn.Module):
    """Simple PyTorch MLP for flattened time-series inputs."""

    def __init__(
        self,
        input_size,
        seq_len,
        n_classes,
        hidden_dim,
        n_layers,
        dropout,
        activation_hidden,
        activation,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        hidden_activation = _get_activation_module(activation_hidden)
        output_activation = _get_activation_module(activation)
        dropout_values = _normalize_dropout(dropout, n_layers)

        layers = []
        in_features = input_size * seq_len
        for layer_idx in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            if hidden_activation is not None:
                layers.append(hidden_activation())
            if dropout_values[layer_idx] > 0:
                layers.append(nn.Dropout(dropout_values[layer_idx]))
            in_features = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, n_classes)
        self.output_activation = (
            None if output_activation is None else output_activation()
        )

    def forward(self, X):
        import torch.nn.functional as F

        if X.ndim != 3:
            raise ValueError(
                "Expected input shape (batch, timesteps, channels) or "
                "(batch, channels, timesteps)."
            )

        if X.shape[2] == self.input_size:
            pass
        elif X.shape[1] == self.input_size:
            X = X.transpose(1, 2)
        else:
            raise ValueError(
                "Unexpected input shape "
                f"{tuple(X.shape)} for input_size={self.input_size}, "
                f"seq_len={self.seq_len}."
            )

        if X.shape[1] != self.seq_len:
            # The notebooks evaluate on synthetic streams and fixed-size windows
            # that do not match the FordA training length. Resample along time
            # so the classifier can score those inputs consistently.
            X = F.interpolate(
                X.transpose(1, 2),
                size=self.seq_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        X = X.reshape(X.shape[0], -1)
        X = self.feature_extractor(X)
        X = self.classifier(X)
        if self.output_activation is not None:
            X = self.output_activation(X)
        return X


class MLPClassifierTorch(BaseDeepClassifierPytorch):
    """Project-local PyTorch MLP compatible with current sktime releases."""

    _tags = {
        "authors": ["OpenAI"],
        "maintainers": ["OpenAI"],
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float | Sequence[float] = 0.0,
        num_epochs: int = 100,
        batch_size: int = 16,
        activation: str | None | Callable = None,
        activation_hidden: str | None | Callable = "relu",
        optimizer: str | None | Callable = "Adam",
        criterion: str | None | Callable = "CrossEntropyLoss",
        callbacks: None | str | tuple[str, ...] = "ReduceLROnPlateau",
        optimizer_kwargs: dict | None = None,
        criterion_kwargs: dict | None = None,
        callback_kwargs: dict | None = None,
        lr: float = 0.001,
        verbose: bool = False,
        random_state: int | None = 0,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation_hidden = activation_hidden

        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            activation=activation,
            criterion=criterion,
            criterion_kwargs=criterion_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            callbacks=callbacks,
            callback_kwargs=callback_kwargs,
            lr=lr,
            verbose=verbose,
            random_state=random_state,
        )

    def _build_network(self, X, y):
        if X.ndim != 3:
            raise ValueError(
                "Expected 3D input X with shape "
                "(n_instances, n_dims, series_length)."
            )

        _, input_size, seq_len = X.shape
        n_classes = len(np.unique(y))
        return ProjectMLPNetworkTorch(
            input_size=input_size,
            seq_len=seq_len,
            n_classes=n_classes,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            activation_hidden=self.activation_hidden,
            activation=self._validated_activation,
        )

    def _fit(self, X, y):
        result = super()._fit(X, y)
        self.network_ = self.network
        return result


def build_mlp_classifier():
    """Build the project MLP classifier with configured hyperparameters."""
    return MLPClassifierTorch(
        hidden_dim=MLP_HIDDEN_DIM,
        n_layers=MLP_N_LAYERS,
        dropout=MLP_DROPOUT,
        num_epochs=MLP_NUM_EPOCHS,
        batch_size=MLP_BATCH_SIZE,
        lr=MLP_LR,
        optimizer=MLP_OPTIMIZER,
        criterion=MLP_CRITERION,
        activation=None,
        activation_hidden="relu",
        callbacks="ReduceLROnPlateau",
        verbose=True,
        random_state=RANDOM_SEED,
    )


def build_tsf_baseline():
    """Build a TimeSeriesForestClassifier baseline."""
    return TimeSeriesForestClassifier(
        n_estimators=200,
        random_state=RANDOM_SEED,
    )


def _normalize_dropout(dropout, n_layers):
    if isinstance(dropout, Sequence) and not isinstance(dropout, (str, bytes)):
        values = [float(value) for value in dropout]
    elif dropout is None:
        values = [0.0]
    else:
        values = [float(dropout)]

    if not values:
        values = [0.0]

    if len(values) < n_layers:
        values.extend([values[-1]] * (n_layers - len(values)))
    else:
        values = values[:n_layers]

    for value in values:
        if not 0.0 <= value < 1.0:
            raise ValueError("Dropout values must be in the range [0, 1).")

    return values


def _get_activation_module(activation):
    if activation is None:
        return None
    if not isinstance(activation, str):
        raise TypeError(
            "Only string activation names or None are supported in this project MLP."
        )

    import torch.nn as nn

    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "elu": nn.ELU,
        "leakyrelu": nn.LeakyReLU,
        "leaky_relu": nn.LeakyReLU,
        "logsoftmax": lambda: nn.LogSoftmax(dim=-1),
        "softmax": lambda: nn.Softmax(dim=-1),
    }

    key = activation.lower()
    if key not in activations:
        raise ValueError(f"Unsupported activation: {activation}")
    return activations[key]
