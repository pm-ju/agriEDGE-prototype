# agriEDGE prototype

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

## Project Structure

- `dashboard/` Streamlit review app
- `src/` model builders, detector logic, evaluation helpers, and ONNX export
- `data/synthetic/` stored synthetic signals and the generator script
- `models/` exported model artifacts used by the dashboard
- `notebooks/` exploration, training, evaluation, explainability, and export

## Notes

- This is a prototype, not a production detector.
- The classifier is trained on FordA as proxy data, not on field recordings from a harvester.
- The dashboard replays stored 10 second samples; it is not connected to a live sensor feed.
- The event marker shown in the dashboard is the injected event start saved with the synthetic sample.

## Future Work

- Train on field data instead of the FordA proxy dataset
- Connect the dashboard to the chunked stream detector path
- Add tests around window scoring and event merging
- Compare the batch dashboard path with the stateful stream API
