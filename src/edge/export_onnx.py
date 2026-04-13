"""Export trained sktime PyTorch models to ONNX for embedded deployment.

This module handles the critical "edge AI" aspect:
- Extract the underlying torch.nn.Module from the sktime estimator
- Export it to ONNX format (industry standard for embedded inference)
- Apply dynamic quantization (INT8) to reduce model size and latency
- Benchmark inference time comparing PyTorch vs ONNX vs Quantized ONNX

This proves the model can run on an agricultural ECU or embedded Jetson device.
"""

import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def extract_torch_model(sktime_estimator):
    """Extract the underlying PyTorch nn.Module from an sktime estimator.

    Parameters
    ----------
    sktime_estimator : BaseDeepClassifierPytorch or BaseDeepRegressorTorch
        A fitted sktime deep learning estimator.

    Returns
    -------
    model : torch.nn.Module
        The underlying PyTorch neural network.
    """
    # Current sktime PyTorch estimators may expose either network_ or network.
    if hasattr(sktime_estimator, "network_"):
        return sktime_estimator.network_
    if hasattr(sktime_estimator, "network"):
        return sktime_estimator.network
    raise AttributeError(
        "Could not find the underlying PyTorch model. "
        "Make sure the estimator has been fitted first."
    )


def export_to_onnx(model, input_shape, output_path, opset_version=17):
    """Export a PyTorch model to ONNX format.

    Parameters
    ----------
    model : torch.nn.Module
        The model to export.
    input_shape : tuple
        Shape of a single input batch, e.g., (1, 1, 500) for
        (batch=1, channels=1, timesteps=500).
    output_path : str or Path
        Where to save the .onnx file.
    opset_version : int
        ONNX opset version.
    """
    import torch

    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        dynamo=False,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["sensor_input"],
        output_names=["detection_output"],
        dynamic_axes={
            "sensor_input": {0: "batch_size"},
            "detection_output": {0: "batch_size"},
        },
    )
    print(f"ONNX model saved to: {output_path}")


def quantize_onnx(input_path, output_path):
    """Apply dynamic INT8 quantization to an ONNX model.

    Parameters
    ----------
    input_path : str or Path
        Path to the original .onnx file.
    output_path : str or Path
        Where to save the quantized .onnx file.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    temp_dir = Path(output_path).parent / "_tmp_quant"
    temp_dir.mkdir(parents=True, exist_ok=True)

    with _project_tempdir(temp_dir):
        quantize_dynamic(
            str(input_path),
            str(output_path),
            weight_type=QuantType.QUInt8,
        )
    print(f"Quantized ONNX model saved to: {output_path}")


def benchmark_inference(model_paths, input_shape, n_iterations=1000):
    """Benchmark inference latency across PyTorch, ONNX, and Quantized ONNX.

    Parameters
    ----------
    model_paths : dict
        Keys: "pytorch" (torch.nn.Module), "onnx" (path), "quantized" (path).
    input_shape : tuple
        e.g., (1, 1, 500)
    n_iterations : int
        Number of inference iterations for timing.

    Returns
    -------
    results : dict
        Keys are model names, values are dicts with:
        - "mean_latency_ms": average inference time
        - "std_latency_ms": standard deviation
        - "model_size_kb": file size in kilobytes
    """
    import os

    import onnxruntime as ort
    import torch

    results = {}

    # --- PyTorch benchmark ---
    if "pytorch" in model_paths:
        torch_model = model_paths["pytorch"]
        torch_model.eval()
        dummy = torch.randn(*input_shape)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = torch_model(dummy)

        latencies = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = torch_model(dummy)
            latencies.append((time.perf_counter() - start) * 1000)

        results["PyTorch (FP32)"] = {
            "mean_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "model_size_kb": "N/A (in-memory)",
        }

    # --- ONNX benchmark ---
    for name, path_key in [("ONNX (FP32)", "onnx"), ("ONNX (INT8)", "quantized")]:
        if path_key in model_paths:
            path = str(model_paths[path_key])
            session = ort.InferenceSession(path)
            dummy = np.random.randn(*input_shape).astype(np.float32)
            input_name = session.get_inputs()[0].name

            # Warmup
            for _ in range(10):
                session.run(None, {input_name: dummy})

            latencies = []
            for _ in range(n_iterations):
                start = time.perf_counter()
                session.run(None, {input_name: dummy})
                latencies.append((time.perf_counter() - start) * 1000)

            results[name] = {
                "mean_latency_ms": np.mean(latencies),
                "std_latency_ms": np.std(latencies),
                "model_size_kb": os.path.getsize(path) / 1024,
            }

    return results


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
    """Create temp directories without relying on broken default temp ACLs."""
    base_dir = Path(dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    while True:
        candidate = base_dir / f"{prefix}{uuid.uuid4().hex}{suffix}"
        try:
            candidate.mkdir()
            return str(candidate)
        except FileExistsError:
            continue
