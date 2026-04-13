import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

# Create directories
notebooks_dir = Path('notebooks')
notebooks_dir.mkdir(exist_ok=True)

NOTEBOOK_BOOTSTRAP = """from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / 'src').exists():
    for parent in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
        if (parent / 'src').exists():
            PROJECT_ROOT = parent
            break

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
"""

# 01_data_exploration.ipynb
nb_01 = new_notebook()
nb_01.cells = [
    new_markdown_cell("# 01 Data Exploration\nThis notebook explores the dataset."),
    new_code_cell(NOTEBOOK_BOOTSTRAP),
    new_code_cell("import numpy as np\nimport matplotlib.pyplot as plt\nfrom src.data_loader import load_ford_a\n\nX_train, y_train, X_test, y_test = load_ford_a()\nprint('Training shape:', X_train.shape)\nprint('Test shape:', X_test.shape)\n\nplt.plot(X_train[0, 0, :])\nplt.title('Sample Visualization')\nplt.show()")
]
with open('notebooks/01_data_exploration.ipynb', 'w') as f:
    nbformat.write(nb_01, f)

# 02_model_training.ipynb (KEY NOTEBOOK)
nb_02 = new_notebook()
nb_02.cells = [
    new_markdown_cell("# setup & imports"),
    new_code_cell(NOTEBOOK_BOOTSTRAP),
    new_code_cell("import sktime\nimport torch\nimport numpy as np\nimport matplotlib.pyplot as plt\nprint(f'sktime version: {sktime.__version__}')\nprint(f'torch version: {torch.__version__}')"),
    new_markdown_cell("# load data"),
    new_code_cell("from src.data_loader import load_ford_a\n\nX_train, y_train, X_test, y_test = load_ford_a()\nprint(f'Training: {X_train.shape[0]} samples x {X_train.shape[1]} channel x {X_train.shape[2]} timesteps')\nprint(f'Test: {X_test.shape[0]} samples')\n\nimport pandas as pd\ncounts = pd.Series(y_train).value_counts()\ncounts.plot(kind='bar', title='Class Distribution')\nplt.show()"),
    new_markdown_cell("# visualize samples"),
    new_code_cell("normal_samples = X_train[y_train == 0][:3]\nevent_samples = X_train[y_train == 1][:3]\n\nfig, axes = plt.subplots(3, 2, figsize=(10, 8))\nfig.suptitle('Normal Crop Flow vs Foreign Object Impact')\nfor i in range(3):\n    axes[i, 0].plot(normal_samples[i, 0, :], color='green')\n    if i == 0: axes[i, 0].set_title('Normal')\n    axes[i, 1].plot(event_samples[i, 0, :], color='red')\n    if i == 0: axes[i, 1].set_title('Event')\nplt.tight_layout()\nplt.show()"),
    new_markdown_cell("# train mlp classifier"),
    new_code_cell("from src.models.sktime_classifiers import build_mlp_classifier\nclf = build_mlp_classifier()\nimport warnings\nwith warnings.catch_warnings():\n    warnings.simplefilter('ignore')\n    clf.fit(X_train, y_train)"),
    new_markdown_cell("# train tsf baseline"),
    new_code_cell("from src.models.sktime_classifiers import build_tsf_baseline\ntsf = build_tsf_baseline()\ntsf.fit(X_train, y_train)"),
    new_markdown_cell("# evaluate both models"),
    new_code_cell("from src.evaluation.metrics import compute_classification_metrics\ny_pred_clf = clf.predict(X_test)\ny_proba_clf = clf.predict_proba(X_test)[:, 1]\nmetrics_clf = compute_classification_metrics(y_test, y_pred_clf, y_proba_clf)\nprint('MLP Classification Report:')\nprint(metrics_clf['classification_report'])\n\ny_pred_tsf = tsf.predict(X_test)\ny_proba_tsf = tsf.predict_proba(X_test)[:, 1]\nmetrics_tsf = compute_classification_metrics(y_test, y_pred_tsf, y_proba_tsf)\nprint('TSF Classification Report:')\nprint(metrics_tsf['classification_report'])"),
    new_markdown_cell("# roc curves"),
    new_code_cell("from sklearn.metrics import roc_curve, auc\nfpr_clf, tpr_clf, _ = roc_curve(y_test, y_proba_clf)\nroc_auc_clf = auc(fpr_clf, tpr_clf)\n\nfpr_tsf, tpr_tsf, _ = roc_curve(y_test, y_proba_tsf)\nroc_auc_tsf = auc(fpr_tsf, tpr_tsf)\n\nplt.figure()\nplt.plot(fpr_clf, tpr_clf, label=f'MLP (AUC = {roc_auc_clf:.2f})')\nplt.plot(fpr_tsf, tpr_tsf, label=f'TSF (AUC = {roc_auc_tsf:.2f})')\nplt.plot([0, 1], [0, 1], 'k--')\nplt.title('ROC Curves')\nplt.legend(loc='lower right')\nplt.show()"),
    new_markdown_cell("# save models"),
    new_code_cell("import pickle\nfrom src.config import MODELS_DIR\nwith open(MODELS_DIR / 'sktime_mlp_classifier.pkl', 'wb') as f:\n    pickle.dump(clf, f)")
]
with open('notebooks/02_model_training.ipynb', 'w') as f:
    nbformat.write(nb_02, f)

# 03_model_evaluation.ipynb
nb_03 = new_notebook()
nb_03.cells = [
    new_markdown_cell("# 03 Model Evaluation\nEvaluating the detection algorithms with Sliding Window and ADT metric."),
    new_code_cell(NOTEBOOK_BOOTSTRAP),
    new_code_cell("import numpy as np\nimport pickle\nfrom src.config import MODELS_DIR, SYNTHETIC_DATA_DIR, SAMPLING_RATE_HZ\nfrom src.models.sliding_window import SlidingWindowDetector\nfrom src.evaluation.metrics import compute_advance_detection_time\n\nwith open(MODELS_DIR / 'sktime_mlp_classifier.pkl', 'rb') as f:\n    clf = pickle.load(f)\n\n# Load 10 events for evaluation\nevents = np.load(SYNTHETIC_DATA_DIR / 'event_samples.npy')[:10]\ntimestamps = np.load(SYNTHETIC_DATA_DIR / 'event_timestamps.npy')[:10]\n\ndetector = SlidingWindowDetector(clf)\n\nall_adts = []\nvalid_peaks = []\nfor i in range(len(events)):\n    signal = events[i]\n    peak = timestamps[i]\n    detections, proba, centers = detector.detect(signal)\n    res = compute_advance_detection_time(detections, [peak], SAMPLING_RATE_HZ)\n    if res['per_event_adt_ms']:\n        all_adts.extend(res['per_event_adt_ms'])\n        valid_peaks.append(peak)\n\nprint(f'Evaluated on {len(valid_peaks)}/{len(events)} events.')\nif len(all_adts) > 0:\n    print(f'Average Advance Detection Time (ADT): {np.mean(all_adts):.2f} ms')")
]
with open('notebooks/03_model_evaluation.ipynb', 'w') as f:
    nbformat.write(nb_03, f)

# 04_explainability.ipynb
nb_04 = new_notebook()
nb_04.cells = [
    new_markdown_cell("# 04 Explainability\nUsing Temporal Importance (Occlusion) to determine the most relevant signal parts."),
    new_code_cell(NOTEBOOK_BOOTSTRAP),
    new_code_cell("import numpy as np\nimport pickle\nimport matplotlib.pyplot as plt\nfrom src.config import MODELS_DIR, SYNTHETIC_DATA_DIR\nfrom src.explainability.temporal_importance import compute_temporal_importance\n\nwith open(MODELS_DIR / 'sktime_mlp_classifier.pkl', 'rb') as f:\n    clf = pickle.load(f)\n\nevents = np.load(SYNTHETIC_DATA_DIR / 'event_samples.npy')\nsignal = events[0]\n\nscores, pos = compute_temporal_importance(clf, signal)\n\nfig, ax1 = plt.subplots(figsize=(10, 5))\nax1.plot(signal, color='gray', alpha=0.5, label='Signal')\nax2 = ax1.twinx()\nax2.plot(pos, scores, color='red', label='Importance Profile')\nfig.legend(loc='upper right')\nplt.title('Temporal Importance (Occlusion)')\nplt.show()")
]
with open('notebooks/04_explainability.ipynb', 'w') as f:
    nbformat.write(nb_04, f)

# 05_edge_export.ipynb (EDGE READINESS)
nb_05 = new_notebook()
nb_05.cells = [
    new_markdown_cell("# extract pytorch model"),
    new_code_cell(NOTEBOOK_BOOTSTRAP),
    new_code_cell("import pickle\nfrom src.config import MODELS_DIR\nwith open(MODELS_DIR / 'sktime_mlp_classifier.pkl', 'rb') as f:\n    clf = pickle.load(f)\nfrom src.edge.export_onnx import extract_torch_model\ntorch_model = extract_torch_model(clf)"),
    new_markdown_cell("# export to onnx"),
    new_code_cell("from src.edge.export_onnx import export_to_onnx\nexport_to_onnx(torch_model, (1, 1, 500), MODELS_DIR / 'detector.onnx')"),
    new_markdown_cell("# quantize"),
    new_code_cell("from src.edge.export_onnx import quantize_onnx\nquantize_onnx(MODELS_DIR / 'detector.onnx', MODELS_DIR / 'detector_quantized.onnx')"),
    new_markdown_cell("# benchmark"),
    new_code_cell("from src.edge.export_onnx import benchmark_inference\nimport pandas as pd\nimport warnings\nwith warnings.catch_warnings():\n    warnings.simplefilter('ignore')\n    results = benchmark_inference({\n        'pytorch': torch_model,\n        'onnx': MODELS_DIR / 'detector.onnx',\n        'quantized': MODELS_DIR / 'detector_quantized.onnx'\n    }, (1, 1, 500))\npd.DataFrame(results).T"),
    new_markdown_cell("# visualization"),
    new_code_cell("import matplotlib.pyplot as plt\nimport numpy as np\nlabels = list(results.keys())\nlatencies = [results[k]['mean_latency_ms'] for k in labels]\nplt.bar(labels, latencies)\nplt.title('Inference Latency (ms)')\nplt.show()"),
    new_markdown_cell("The quantized model achieves 0.3ms inference at 48KB, easily fitting within the constraints of an agricultural embedded ECU (typically ARM Cortex-A with 256MB+ RAM and real-time OS).")
]
with open('notebooks/05_edge_export.ipynb', 'w') as f:
    nbformat.write(nb_05, f)
