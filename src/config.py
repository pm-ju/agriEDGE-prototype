"""Central configuration for AgriEdge-Detect.

All hyperparameters, paths, and constants live here.
Every other file imports from this module — never hardcodes values.
"""

from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Create dirs if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR,
          MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATA PARAMETERS
# ============================================================
# We use the FordA dataset from sktime as our proxy for vibration data.
# FordA: 3601 train + 1320 test samples, each 500 time steps, binary class.
# Class 1 = "event detected" (foreign object), Class -1 = "normal operation"
DATASET_NAME = "FordA"
RANDOM_SEED = 42

# Synthetic data parameters (for the dashboard simulator)
SAMPLING_RATE_HZ = 1000          # 1 kHz (millisecond granularity, matching real project)
SIGNAL_DURATION_SEC = 10         # Each synthetic sample is 10 seconds
NOISE_AMPLITUDE = 0.05           # Background vibration noise level
EVENT_AMPLITUDE_RANGE = (0.3, 0.8)  # Foreign object impact amplitude range
EVENT_DURATION_MS = (20, 100)    # Impact event duration in milliseconds

# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================
# --- sktime MLP Classifier (Torch) ---
MLP_HIDDEN_DIM = 128
MLP_N_LAYERS = 3
MLP_DROPOUT = (0.1, 0.2, 0.2)
MLP_NUM_EPOCHS = 100
MLP_BATCH_SIZE = 16
MLP_LR = 0.001
MLP_OPTIMIZER = "Adam"
MLP_CRITERION = "CrossEntropyLoss"

# --- Sliding Window Detection ---
WINDOW_SIZE = 50                 # Number of time steps per detection window
WINDOW_STRIDE = 10               # Stride between consecutive windows
DETECTION_THRESHOLD = 0.5        # Probability threshold for triggering alert

# ============================================================
# EDGE / ONNX PARAMETERS
# ============================================================
ONNX_OPSET_VERSION = 17
QUANTIZATION_TYPE = "dynamic"    # "dynamic" or "static"

# ============================================================
# DASHBOARD PARAMETERS
# ============================================================
STREAM_UPDATE_INTERVAL_MS = 50   # Dashboard refresh rate (20 FPS)
CHART_VISIBLE_WINDOW_SEC = 5     # How many seconds of data visible at once
ALERT_COOLDOWN_SEC = 2.0         # Minimum time between consecutive alerts
