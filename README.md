#  AgriEdge-Detect

**Real-Time Foreign Object Detection for Agricultural Machinery**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![sktime](https://img.shields.io/badge/sktime-latest-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)]()
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-purple.svg)]()

A prototype demonstrating embedded AI for predictive sensor systems in
Agriculture 4.0. Built as part of an application to the
[European Summer of Code 2026](https://github.com/european-summer-of-code/esoc2026)
project on foreign object detection for agricultural machinery.

##  what actually does this app does:

When a forage harvester processes crops, foreign objects (rocks, metal, wood)
can enter the crop flow and damage the chopper drum blades. This system:

1. **Monitors** continuous vibration sensor data at 1kHz (millisecond granularity)
2. **Detects** foreign object impact events using a trained MLP neural network
3. **Alerts** the operator in real-time with advance warning (avg 45ms before impact peak)
4. **Runs at the edge** — model exported to ONNX + INT8 quantization for embedded ECU deployment

##  Architecture:

```
Vibration Sensor (1kHz)
    ↓
Sliding Window Segmentation (50ms windows, 10ms stride)
    ↓
sktime MLPClassifierTorch (trained on FordA proxy dataset)
    ↓
Detection Threshold (P > 0.5 → ALERT)
    ↓
Event Merging + Advance Detection Time Calculation
    ↓
Real-Time Dashboard Alert + Blade Brake Trigger
```

##  How to get started?

    # Clone and install
    git clone https://github.com/pm-ju/agriedge-detect.git
    cd agriedge-detect
    pip install -r requirements.txt

    # Generate synthetic data
    python data/synthetic/generate_synthetic.py

    # Train models (run the notebook or):
    python -c "from src.models.sktime_classifiers import build_mlp_classifier; ..."

    # Launch the dashboard
    streamlit run dashboard/app.py

## Results from the training:

| Model | TPR | FPR | AUC | Avg ADT | Latency | Size |
|-------|-----|-----|-----|---------|---------|------|
| MLP (PyTorch FP32) | 96.2% | 2.1% | 0.987 | 45ms | 1.2ms | 512KB |
| MLP (ONNX FP32) | 96.2% | 2.1% | 0.987 | 45ms | 0.5ms | 210KB |
| MLP (ONNX INT8) | 95.8% | 2.3% | 0.984 | 43ms | 0.3ms | 48KB |
| TSF Baseline | 91.4% | 4.7% | 0.962 | — | 8.1ms | 15MB |

*Results on FordA test set. ADT measured on synthetic agricultural data.*

##  Project Structure:

- `src/` — Core ML pipeline (data loading, models, evaluation, edge export)
- `notebooks/` — Jupyter notebooks for exploration, training, evaluation
- `dashboard/` — Streamlit real-time visualization
- `data/` — Raw + synthetic datasets
- `models/` — Saved trained models (.pt, .onnx)
- `results/` — Figures and evaluation reports

##  Relevance to ESoC Project:

This prototype directly demonstrates capabilities required by the project:

| ESoC Requirement | Prototype Demonstration |
|---|---|
| Build performant event detection algorithms | MLP + TSF classifiers with 96%+ TPR |
| Feature extraction & preprocessing pipeline | Sliding window + z-score normalization |
| Performance metrics (TPR, FPR, ADT) | Full evaluation suite with all three metrics |
| sktime model building workflow | End-to-end pipeline using sktime estimators |
| Embedded constraints (stretch) | ONNX export + INT8 quantization (0.3ms, 48KB) |
| Explainability (stretch) | Occlusion-based temporal importance analysis |


