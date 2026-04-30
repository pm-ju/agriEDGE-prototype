# ESOC Prototype project

A Streamlit prototype for reviewing sliding-window event detection on synthetic harvester vibration data.

Live app: https://agri-edge.streamlit.app/

## Features

- Replay stored normal and injected-event signals
- Score fixed windows with a simple heuristic and an exported ONNX model
- Merge windows above a threshold into detection regions
- Inspect the signal trace, score trace, and recent detections in one dashboard
- Keep the training and export workflow in notebooks for easy iteration

## Stack

- Python
- Streamlit
- sktime
- PyTorch
- ONNX Runtime
- NumPy
- Plotly

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

If you want to regenerate the synthetic samples:

```bash
python data/synthetic/generate_synthetic.py
```

If the ONNX model is missing, export it from the notebook:

```bash
jupyter notebook notebooks/05_edge_export.ipynb
```
