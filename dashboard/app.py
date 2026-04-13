"""AgriEdge-Detect dashboard with live synthetic playback and alert visualization."""

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    ALERT_COOLDOWN_SEC,
    CHART_VISIBLE_WINDOW_SEC,
    MODELS_DIR,
    SAMPLING_RATE_HZ,
    STREAM_UPDATE_INTERVAL_MS,
    SYNTHETIC_DATA_DIR,
    WINDOW_SIZE,
    WINDOW_STRIDE,
)


APP_TITLE = "AgriEdge-Detect"
TARGET_SEQUENCE_LENGTH = 500
REFRESH_INTERVAL_SEC = max(0.12, STREAM_UPDATE_INTERVAL_MS / 1000 * 3)
DEFAULT_EVENT_SAMPLE = 2
DEFAULT_NORMAL_SAMPLE = 0
EVENT_SPAN_MS = 80

SIGNAL_EVENT = "Foreign object event"
SIGNAL_NORMAL = "Nominal crop flow"
ENGINE_ADAPTIVE = "Adaptive impact demo"
ENGINE_COMPOSITE = "Composite demo"
ENGINE_MODEL = "Edge model only"


def create_layout():
    """Configure the Streamlit page and the custom visual system."""
    st.set_page_config(
        page_title=f"{APP_TITLE} | Live Monitor",
        page_icon="A",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
            :root {
                --bg: #0b1010;
                --panel: rgba(16, 25, 25, 0.88);
                --panel-border: rgba(212, 232, 185, 0.12);
                --text: #f3efe1;
                --muted: #97a79f;
                --lime: #c8ff63;
                --mint: #61f0b3;
                --amber: #ffb454;
                --red: #ff6a4d;
                --blue: #67bfff;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(200, 255, 99, 0.08), transparent 30%),
                    radial-gradient(circle at top right, rgba(255, 180, 84, 0.08), transparent 28%),
                    linear-gradient(180deg, #081011 0%, #101918 48%, #0d1614 100%);
                color: var(--text);
            }

            [data-testid="stHeader"] {
                background: rgba(7, 11, 11, 0.65);
                border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(15, 22, 22, 0.97), rgba(10, 15, 15, 0.97));
                border-left: 1px solid rgba(255, 255, 255, 0.06);
            }

            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2rem;
                max-width: 1500px;
            }

            .hero-shell {
                background:
                    linear-gradient(135deg, rgba(20, 31, 29, 0.96), rgba(14, 22, 20, 0.96)),
                    linear-gradient(90deg, rgba(200, 255, 99, 0.08), rgba(255, 180, 84, 0.08));
                border: 1px solid var(--panel-border);
                border-radius: 28px;
                padding: 1.6rem;
                box-shadow: 0 28px 90px rgba(0, 0, 0, 0.24);
                overflow: hidden;
                position: relative;
            }

            .hero-shell:before {
                content: "";
                position: absolute;
                inset: 0;
                background:
                    linear-gradient(120deg, rgba(255, 255, 255, 0.05), transparent 25%),
                    radial-gradient(circle at 82% 18%, rgba(200, 255, 99, 0.16), transparent 18%);
                pointer-events: none;
            }

            .hero-kicker,
            .section-label,
            [data-testid="stMetricLabel"] {
                font-family: "Trebuchet MS", "Segoe UI Variable", sans-serif;
                text-transform: uppercase;
                color: var(--muted);
                letter-spacing: 0.13em;
            }

            .hero-kicker {
                color: var(--amber);
                font-size: 0.76rem;
                margin-bottom: 0.8rem;
            }

            .hero-title,
            .panel-title,
            .hero-stat-value,
            .kpi-value,
            [data-testid="stMetricValue"] {
                font-family: "Georgia", "Palatino Linotype", serif;
                color: var(--text);
            }

            .hero-title {
                font-size: 3.35rem;
                line-height: 0.96;
                margin: 0;
            }

            .hero-accent {
                color: var(--lime);
            }

            .hero-copy,
            .panel-copy,
            .mini-note,
            .log-entry,
            .footer-note {
                font-family: "Trebuchet MS", "Segoe UI Variable", sans-serif;
            }

            .hero-copy {
                color: #d4ddd7;
                font-size: 1.08rem;
                max-width: 46rem;
                margin: 0.95rem 0 1.2rem 0;
            }

            .hero-badges {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
            }

            .hero-badge {
                display: inline-flex;
                align-items: center;
                padding: 0.44rem 0.72rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.08);
                font-family: "Trebuchet MS", "Segoe UI Variable", sans-serif;
                font-size: 0.9rem;
                color: var(--text);
            }

            .hero-rail {
                display: grid;
                gap: 0.8rem;
                margin-top: 0.15rem;
            }

            .hero-stat,
            .kpi-card,
            [data-testid="stMetric"],
            .log-entry {
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }

            .hero-stat {
                border-radius: 22px;
                padding: 0.9rem 1rem;
            }

            .hero-stat-label,
            .kpi-label {
                font-family: "Trebuchet MS", "Segoe UI Variable", sans-serif;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.13em;
                font-size: 0.68rem;
                margin-bottom: 0.34rem;
            }

            .hero-stat-value {
                font-size: 1.75rem;
                line-height: 1.05;
            }

            .toolbar-shell,
            .panel-shell {
                background: rgba(12, 20, 19, 0.84);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 24px;
            }

            .toolbar-shell {
                margin-top: 1rem;
                padding: 1rem 1rem 0.2rem 1rem;
            }

            .section-label {
                font-size: 0.72rem;
                margin-bottom: 0.3rem;
            }

            .panel-shell {
                padding: 0.8rem 1rem 1rem 1rem;
                box-shadow: 0 16px 48px rgba(0, 0, 0, 0.18);
                height: 100%;
            }

            .panel-title {
                font-size: 1.5rem;
                margin-bottom: 0.15rem;
            }

            .panel-copy {
                color: var(--muted);
                font-size: 0.92rem;
                margin-bottom: 0.9rem;
            }

            .status-banner {
                padding: 1rem 1.1rem;
                border-radius: 22px;
                border: 1px solid rgba(255, 255, 255, 0.06);
                font-family: "Trebuchet MS", "Segoe UI Variable", sans-serif;
                font-weight: 700;
                font-size: 1rem;
                margin-bottom: 0.95rem;
            }

            .status-ok {
                background: linear-gradient(90deg, rgba(36, 72, 54, 0.92), rgba(17, 39, 31, 0.92));
                color: #dfffe5;
                box-shadow: 0 0 0 1px rgba(97, 240, 179, 0.12) inset;
            }

            .status-alert {
                background: linear-gradient(90deg, rgba(104, 30, 20, 0.96), rgba(70, 16, 10, 0.96));
                color: #ffe7df;
                box-shadow: 0 0 0 1px rgba(255, 106, 77, 0.18) inset, 0 0 35px rgba(255, 106, 77, 0.14);
            }

            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.75rem;
                margin-bottom: 1rem;
            }

            .kpi-card {
                border-radius: 20px;
                padding: 0.9rem 0.95rem;
            }

            .kpi-value {
                font-size: 1.55rem;
                line-height: 1.05;
            }

            .mini-note {
                color: #c3d2cc;
                font-size: 0.92rem;
                padding: 0.92rem 1rem;
                border-radius: 18px;
                background: linear-gradient(135deg, rgba(28, 43, 56, 0.95), rgba(18, 30, 40, 0.95));
                border: 1px solid rgba(103, 191, 255, 0.12);
                margin-top: 0.85rem;
            }

            .log-entry {
                padding: 0.7rem 0.8rem;
                border-radius: 16px;
                margin-bottom: 0.55rem;
            }

            .log-title {
                color: var(--text);
                font-weight: 700;
                margin-bottom: 0.18rem;
            }

            .log-copy,
            .footer-note {
                color: var(--muted);
                font-size: 0.9rem;
            }

            .stButton button {
                border-radius: 999px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                background: linear-gradient(135deg, rgba(24, 32, 31, 0.98), rgba(14, 20, 20, 0.98));
                color: var(--text);
                font-family: "Trebuchet MS", "Segoe UI Variable", sans-serif;
                font-weight: 700;
                min-height: 2.85rem;
            }

            .stButton button:hover {
                border-color: rgba(200, 255, 99, 0.34);
                color: var(--lime);
            }

            [data-testid="stMetric"] {
                border-radius: 18px;
                padding: 0.8rem 0.95rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_signal_bank():
    """Load synthetic normal and event signals for the dashboard."""
    return {
        "normal": np.load(SYNTHETIC_DATA_DIR / "normal_samples.npy").astype(np.float32),
        "event": np.load(SYNTHETIC_DATA_DIR / "event_samples.npy").astype(np.float32),
        "event_timestamps": np.load(SYNTHETIC_DATA_DIR / "event_timestamps.npy").astype(
            np.int64
        ),
    }


@st.cache_resource(show_spinner=False)
def load_edge_runtime():
    """Load the fastest available ONNX runtime session."""
    candidate_paths = [
        MODELS_DIR / "detector_quantized.onnx",
        MODELS_DIR / "detector.onnx",
    ]
    model_path = next((path for path in candidate_paths if path.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            "No exported ONNX model was found. Run notebook 05_edge_export.ipynb first."
        )

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    return session, model_path


@st.cache_data(show_spinner=False)
def measure_edge_runtime(model_path_str):
    """Measure rough edge inference stats once per model file."""
    model_path = Path(model_path_str)
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(256, 1, TARGET_SEQUENCE_LENGTH).astype(np.float32)

    for _ in range(8):
        session.run(None, {input_name: dummy})

    latencies = []
    for _ in range(20):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - start) * 1000)

    batch_latency_ms = float(np.mean(latencies))
    return {
        "engine_name": model_path.name,
        "model_size_kb": model_path.stat().st_size / 1024,
        "batch_latency_ms": batch_latency_ms,
        "single_window_ms": batch_latency_ms / dummy.shape[0],
    }


def initialize_state():
    """Initialize the session state used by the dashboard."""
    defaults = {
        "signal_family": SIGNAL_EVENT,
        "sample_number": DEFAULT_EVENT_SAMPLE + 1,
        "alert_engine": ENGINE_ADAPTIVE,
        "alert_threshold": 0.70,
        "playback_speed": 1.0,
        "playback_cursor": 0,
        "playback_running": True,
        "loop_playback": True,
        "active_signal_token": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_selected_signal(signal_bank):
    """Return the selected signal and the synthetic event timestamp if present."""
    signal_family = st.session_state.signal_family
    if signal_family == SIGNAL_EVENT:
        signal = signal_bank["event"][st.session_state.sample_number - 1]
        event_timestamp = int(
            signal_bank["event_timestamps"][st.session_state.sample_number - 1]
        )
        signal_kind = "event"
    else:
        signal = signal_bank["normal"][st.session_state.sample_number - 1]
        event_timestamp = None
        signal_kind = "normal"
    return signal, event_timestamp, signal_kind


def build_windows(signal, window_size, stride):
    """Create stepped overlapping windows from a 1D signal."""
    windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)[::stride]
    starts = np.arange(0, len(signal) - window_size + 1, stride)
    centers = starts + window_size // 2
    return windows.astype(np.float32), starts.astype(np.int32), centers.astype(np.int32)


def resample_windows(windows, target_length):
    """Resample each window to the target model length."""
    old_axis = np.linspace(0.0, 1.0, windows.shape[1], dtype=np.float32)
    new_axis = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    return np.stack(
        [np.interp(new_axis, old_axis, row).astype(np.float32) for row in windows],
        axis=0,
    )


def softmax(logits):
    """Compute softmax probabilities from ONNX logits."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def compute_model_scores(windows, session):
    """Run ONNX inference on the stepped windows and return class-1 probability."""
    input_name = session.get_inputs()[0].name
    resized = resample_windows(windows, TARGET_SEQUENCE_LENGTH)[:, np.newaxis, :]
    logits = session.run(None, {input_name: resized})[0]
    return softmax(logits)[:, 1].astype(np.float32)


def compute_impact_scores(windows):
    """Compute an adaptive impact score from local slope spikes."""
    raw_impact = np.max(np.abs(np.diff(windows, axis=1)), axis=1)
    baseline_count = min(len(raw_impact), max(40, int(1000 / WINDOW_STRIDE)))
    baseline = raw_impact[:baseline_count]
    center = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - center))) + 1e-6
    robust_z = (raw_impact - center) / (1.4826 * mad)
    score = 1.0 / (1.0 + np.exp(-(robust_z - 5.0) / 1.2))
    return score.astype(np.float32)


def select_alert_scores(engine_name, impact_scores, model_scores):
    """Select the active score series used for alerting."""
    if engine_name == ENGINE_MODEL:
        return model_scores
    if engine_name == ENGINE_COMPOSITE:
        boosted_model = np.clip(model_scores * 3.5, 0.0, 1.0)
        return np.clip(0.82 * impact_scores + 0.18 * boosted_model, 0.0, 1.0)
    return impact_scores


def merge_alert_windows(scores, starts, threshold, window_size):
    """Merge contiguous windows above threshold into detection events."""
    detected = scores >= threshold
    detections = []
    i = 0
    while i < len(detected):
        if not detected[i]:
            i += 1
            continue

        start_idx = i
        peak_confidence = float(scores[i])
        while i + 1 < len(detected) and detected[i + 1]:
            i += 1
            peak_confidence = max(peak_confidence, float(scores[i]))
        end_idx = i

        detections.append(
            {
                "start": int(starts[start_idx]),
                "end": int(starts[end_idx] + window_size),
                "confidence": peak_confidence,
            }
        )
        i += 1

    return detections


def build_signal_analysis(signal, event_timestamp, session):
    """Analyze the full selected signal for live playback."""
    windows, starts, centers = build_windows(signal, WINDOW_SIZE, WINDOW_STRIDE)
    model_scores = compute_model_scores(windows, session)
    impact_scores = compute_impact_scores(windows)
    composite_scores = select_alert_scores(
        ENGINE_COMPOSITE, impact_scores, model_scores
    )
    return {
        "signal": signal,
        "event_timestamp": event_timestamp,
        "windows": windows,
        "starts": starts,
        "centers": centers,
        "model_scores": model_scores,
        "impact_scores": impact_scores,
        "composite_scores": composite_scores,
        "sample_length": len(signal),
    }


def get_score_bundle(analysis, engine_name, threshold):
    """Return the active score series and detections for the chosen engine."""
    active_scores = select_alert_scores(
        engine_name,
        analysis["impact_scores"],
        analysis["model_scores"],
    )
    detections = merge_alert_windows(
        active_scores,
        analysis["starts"],
        threshold,
        WINDOW_SIZE,
    )
    return active_scores, detections


def advance_playback(sample_length):
    """Advance the playback cursor when auto-play is enabled."""
    if not st.session_state.playback_running:
        return

    step = max(
        WINDOW_STRIDE,
        int(SAMPLING_RATE_HZ * REFRESH_INTERVAL_SEC * st.session_state.playback_speed),
    )
    next_cursor = st.session_state.playback_cursor + step
    if next_cursor < sample_length:
        st.session_state.playback_cursor = next_cursor
        return

    if st.session_state.loop_playback:
        st.session_state.playback_cursor = 0
    else:
        st.session_state.playback_cursor = sample_length - 1
        st.session_state.playback_running = False


def get_visible_range(sample_length, cursor):
    """Compute the visible time range for the live charts."""
    visible_points = int(CHART_VISIBLE_WINDOW_SEC * SAMPLING_RATE_HZ)
    end_idx = min(sample_length, max(cursor + 1, visible_points))
    start_idx = max(0, end_idx - visible_points)
    return start_idx, end_idx


def get_current_window_score(centers, scores, cursor):
    """Return the score associated with the current playback position."""
    idx = np.searchsorted(centers, cursor, side="right") - 1
    idx = int(np.clip(idx, 0, len(scores) - 1))
    return float(scores[idx])


def get_active_alert(detections, cursor):
    """Return the active alert for the cursor, with a short display hold."""
    hold_samples = int(ALERT_COOLDOWN_SEC * SAMPLING_RATE_HZ)
    for detection in reversed(detections):
        if detection["start"] <= cursor <= detection["end"] + hold_samples:
            return detection
    return None


def compute_lead_time_ms(detection, event_timestamp):
    """Compute lead time when a detection occurs before the known event timestamp."""
    if detection is None or event_timestamp is None or detection["start"] > event_timestamp:
        return None
    return float(event_timestamp - detection["start"])


def create_signal_chart(signal, start_idx, end_idx, cursor, detections, event_timestamp):
    """Create the live vibration chart for the visible window."""
    visible_signal = signal[start_idx:end_idx]
    x_seconds = np.arange(start_idx, end_idx) / SAMPLING_RATE_HZ
    y_padding = max(0.06, float(np.std(visible_signal)) * 0.45)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_seconds,
            y=visible_signal,
            mode="lines",
            name="Sensor feed",
            line=dict(color="#61f0b3", width=2.0),
            fill="tozeroy",
            fillcolor="rgba(97, 240, 179, 0.12)",
            hovertemplate="t=%{x:.3f}s<br>amp=%{y:.3f}<extra></extra>",
        )
    )

    for detection in detections:
        if detection["end"] < start_idx or detection["start"] > end_idx:
            continue
        fig.add_vrect(
            x0=max(detection["start"], start_idx) / SAMPLING_RATE_HZ,
            x1=min(detection["end"], end_idx) / SAMPLING_RATE_HZ,
            fillcolor="rgba(255, 106, 77, 0.18)",
            line_color="rgba(255, 106, 77, 0.95)",
            line_width=1.5,
            annotation_text=f"Alert {detection['confidence']:.0%}",
            annotation_position="top left",
            annotation_font_color="#ffe7df",
        )

    if event_timestamp is not None and start_idx <= event_timestamp <= end_idx:
        fig.add_vrect(
            x0=event_timestamp / SAMPLING_RATE_HZ,
            x1=(event_timestamp + EVENT_SPAN_MS) / SAMPLING_RATE_HZ,
            fillcolor="rgba(103, 191, 255, 0.14)",
            line_color="rgba(103, 191, 255, 0.92)",
            line_width=1.4,
            annotation_text="Synthetic impact",
            annotation_position="top right",
            annotation_font_color="#c7ecff",
        )

    fig.add_vline(
        x=cursor / SAMPLING_RATE_HZ,
        line_width=2,
        line_dash="dash",
        line_color="#ffb454",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark",
        margin=dict(l=24, r=18, t=16, b=18),
        height=430,
        showlegend=False,
        xaxis=dict(
            title="Time (s)",
            range=[start_idx / SAMPLING_RATE_HZ, end_idx / SAMPLING_RATE_HZ],
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Vibration amplitude",
            range=[
                float(visible_signal.min() - y_padding),
                float(visible_signal.max() + y_padding),
            ],
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
    )
    return fig


def create_score_chart(
    analysis,
    start_idx,
    end_idx,
    cursor,
    alert_scores,
    threshold,
    event_timestamp,
):
    """Create the score timeline chart under the signal feed."""
    mask = (analysis["centers"] >= start_idx) & (analysis["centers"] <= end_idx)
    x_seconds = analysis["centers"][mask] / SAMPLING_RATE_HZ
    model_scores = analysis["model_scores"][mask]
    impact_scores = analysis["impact_scores"][mask]
    alert_slice = alert_scores[mask]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_seconds,
            y=alert_slice,
            mode="lines",
            name="Alert score",
            line=dict(color="#ffb454", width=2.6),
            fill="tozeroy",
            fillcolor="rgba(255, 180, 84, 0.12)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_seconds,
            y=impact_scores,
            mode="lines",
            name="Impact score",
            line=dict(color="#ff6a4d", width=1.4, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_seconds,
            y=model_scores,
            mode="lines",
            name="ONNX model p(event)",
            line=dict(color="#67bfff", width=1.8),
        )
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#f3efe1",
        line_width=1.2,
        annotation_text=f"threshold {threshold:.2f}",
        annotation_font_color="#f3efe1",
    )
    fig.add_vline(
        x=cursor / SAMPLING_RATE_HZ,
        line_width=2,
        line_dash="dash",
        line_color="#ffb454",
    )
    if event_timestamp is not None and start_idx <= event_timestamp <= end_idx:
        fig.add_vline(
            x=event_timestamp / SAMPLING_RATE_HZ,
            line_width=1.5,
            line_color="#67bfff",
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark",
        margin=dict(l=24, r=18, t=16, b=12),
        height=265,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        xaxis=dict(
            title="Time (s)",
            range=[start_idx / SAMPLING_RATE_HZ, end_idx / SAMPLING_RATE_HZ],
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Detection strength",
            range=[0.0, 1.05],
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
    )
    return fig


def format_log_entries(detections, cursor, event_timestamp):
    """Build a short detection log for the side panel."""
    seen = [d for d in detections if d["start"] <= cursor]
    if not seen:
        return []

    entries = []
    for detection in seen[-5:][::-1]:
        lead_time = compute_lead_time_ms(detection, event_timestamp)
        lead_copy = "lead time unavailable" if lead_time is None else f"{lead_time:.0f} ms lead"
        entries.append(
            {
                "title": f"Alert at {detection['start'] / SAMPLING_RATE_HZ:.3f}s",
                "copy": f"confidence {detection['confidence']:.0%} | {lead_copy}",
            }
        )
    return entries


def render_hero(engine_name, signal_family, runtime_stats, cursor_time, active_score):
    """Render the hero section at the top of the page."""
    hero_left, hero_right = st.columns([1.6, 0.95], gap="large")

    with hero_left:
        st.markdown(
            f"""
            <div class="hero-shell">
                <div class="hero-kicker">Predictive sensor cockpit</div>
                <h1 class="hero-title">
                    {APP_TITLE}<br>
                    <span class="hero-accent">Harvester Control Room</span>
                </h1>
                <p class="hero-copy">
                    A live monitor for synthetic harvester vibration streams. It replays a
                    10-second signal, scores each moving window, and raises alerts when the
                    incoming pattern looks like a foreign-object impact.
                </p>
                <div class="hero-badges">
                    <span class="hero-badge">1 kHz feed</span>
                    <span class="hero-badge">{engine_name}</span>
                    <span class="hero-badge">{signal_family}</span>
                    <span class="hero-badge">{runtime_stats['engine_name']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with hero_right:
        st.markdown(
            f"""
            <div class="hero-rail">
                <div class="hero-stat">
                    <div class="hero-stat-label">Playback</div>
                    <div class="hero-stat-value">{cursor_time:.2f}s</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-label">Active score</div>
                    <div class="hero-stat-value">{active_score:.2f}</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-label">Edge latency</div>
                    <div class="hero-stat-value">{runtime_stats['single_window_ms']:.3f} ms</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_toolbar(signal_bank):
    """Render the top-line controls."""
    st.markdown('<div class="toolbar-shell">', unsafe_allow_html=True)
    toolbar_cols = st.columns([1.2, 0.9, 1.1, 0.95, 0.95, 0.55, 0.55, 0.6], gap="medium")

    with toolbar_cols[0]:
        st.markdown('<div class="section-label">Signal source</div>', unsafe_allow_html=True)
        st.selectbox(
            "Signal source",
            [SIGNAL_EVENT, SIGNAL_NORMAL],
            label_visibility="collapsed",
            key="signal_family",
        )

    max_samples = signal_bank["event"].shape[0] if st.session_state.signal_family == SIGNAL_EVENT else signal_bank["normal"].shape[0]
    if st.session_state.signal_family == SIGNAL_EVENT and st.session_state.sample_number == DEFAULT_NORMAL_SAMPLE + 1:
        st.session_state.sample_number = DEFAULT_EVENT_SAMPLE + 1
    if st.session_state.sample_number > max_samples:
        st.session_state.sample_number = 1

    with toolbar_cols[1]:
        st.markdown('<div class="section-label">Sample</div>', unsafe_allow_html=True)
        st.number_input(
            "Sample",
            min_value=1,
            max_value=max_samples,
            step=1,
            label_visibility="collapsed",
            key="sample_number",
        )

    with toolbar_cols[2]:
        st.markdown('<div class="section-label">Alert engine</div>', unsafe_allow_html=True)
        st.selectbox(
            "Alert engine",
            [ENGINE_ADAPTIVE, ENGINE_COMPOSITE, ENGINE_MODEL],
            label_visibility="collapsed",
            key="alert_engine",
        )

    with toolbar_cols[3]:
        st.markdown('<div class="section-label">Threshold</div>', unsafe_allow_html=True)
        st.slider(
            "Threshold",
            min_value=0.15,
            max_value=0.95,
            step=0.01,
            label_visibility="collapsed",
            key="alert_threshold",
        )

    with toolbar_cols[4]:
        st.markdown('<div class="section-label">Speed</div>', unsafe_allow_html=True)
        st.slider(
            "Speed",
            min_value=0.5,
            max_value=3.0,
            step=0.1,
            label_visibility="collapsed",
            key="playback_speed",
        )

    with toolbar_cols[5]:
        st.markdown('<div class="section-label">Run</div>', unsafe_allow_html=True)
        play_label = "Pause" if st.session_state.playback_running else "Play"
        if st.button(play_label, use_container_width=True):
            st.session_state.playback_running = not st.session_state.playback_running

    with toolbar_cols[6]:
        st.markdown('<div class="section-label">Reset</div>', unsafe_allow_html=True)
        if st.button("Reset", use_container_width=True):
            st.session_state.playback_cursor = 0

    with toolbar_cols[7]:
        st.markdown('<div class="section-label">Loop</div>', unsafe_allow_html=True)
        st.toggle(
            "Loop",
            key="loop_playback",
            label_visibility="collapsed",
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_status_and_metrics(
    signal_kind,
    cursor,
    event_timestamp,
    model_score,
    impact_score,
    alert_score,
    detections,
    active_alert,
):
    """Render the right-hand status rail."""
    lead_time = compute_lead_time_ms(active_alert, event_timestamp)
    if active_alert is not None:
        status_class = "status-banner status-alert"
        lead_copy = "lead unavailable" if lead_time is None else f"{lead_time:.0f} ms lead"
        banner_copy = f"ALERT ACTIVE | confidence {active_alert['confidence']:.0%} | {lead_copy}"
    else:
        status_class = "status-banner status-ok"
        banner_copy = (
            "MONITORING | waiting for the injected impact to enter the live window"
            if signal_kind == "event"
            else "MONITORING | nominal crop-flow simulation"
        )

    st.markdown(f'<div class="{status_class}">{banner_copy}</div>', unsafe_allow_html=True)

    total_alerts_seen = len([d for d in detections if d["start"] <= cursor])
    cursor_time = cursor / SAMPLING_RATE_HZ
    event_time_copy = "-" if event_timestamp is None else f"{event_timestamp / SAMPLING_RATE_HZ:.2f}s"

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Cursor</div>
                <div class="kpi-value">{cursor_time:.2f}s</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Alerts seen</div>
                <div class="kpi-value">{total_alerts_seen}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Alert score</div>
                <div class="kpi-value">{alert_score:.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Ground truth impact</div>
                <div class="kpi-value">{event_time_copy}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3)
    metric_cols[0].metric("ONNX model p(event)", f"{model_score:.2f}")
    metric_cols[1].metric("Impact score", f"{impact_score:.2f}")
    metric_cols[2].metric(
        "Lead time",
        "-" if lead_time is None else f"{lead_time:.0f} ms",
    )

    st.markdown(
        """
        <div class="mini-note">
            The dashboard uses the exported ONNX model for the blue trace and an adaptive
            impact detector for the live demo alerts. That keeps the demo responsive and
            visually useful even when the proxy training data does not map cleanly onto
            the synthetic harvester stream.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detection_log(entries):
    """Render the recent detection log."""
    st.markdown('<div class="panel-title">Recent detections</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-copy">Latest alerts that have crossed the chosen threshold.</div>',
        unsafe_allow_html=True,
    )
    if not entries:
        st.markdown(
            """
            <div class="log-entry">
                <div class="log-title">No alert yet</div>
                <div class="log-copy">The live cursor has not crossed a detection window.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for entry in entries:
        st.markdown(
            f"""
            <div class="log-entry">
                <div class="log-title">{entry['title']}</div>
                <div class="log-copy">{entry['copy']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar(runtime_stats, signal_family, engine_name, event_timestamp):
    """Render the technical drawer in the sidebar."""
    with st.sidebar:
        st.markdown("### Technical drawer")
        st.metric("Model file", runtime_stats["engine_name"])
        st.metric("Model size", f"{runtime_stats['model_size_kb']:.1f} KB")
        st.metric("Batch latency", f"{runtime_stats['batch_latency_ms']:.2f} ms")
        st.metric("Single window", f"{runtime_stats['single_window_ms']:.4f} ms")

        st.markdown("---")
        st.markdown("### Current mode")
        st.write(f"Signal: `{signal_family}`")
        st.write(f"Alert engine: `{engine_name}`")
        st.write(f"Threshold: `{st.session_state.alert_threshold:.2f}`")
        if event_timestamp is not None:
            st.write(f"Known event start: `{event_timestamp / SAMPLING_RATE_HZ:.3f}s`")
        else:
            st.write("Known event start: `none`")

        st.markdown("---")
        st.markdown(
            "The original app was mostly static. This rebuilt version renders the "
            "actual signal stream, the ONNX score, the demo alert signal, and the "
            "live playback state."
        )


def main():
    """Run the Streamlit dashboard."""
    create_layout()
    initialize_state()

    try:
        signal_bank = load_signal_bank()
        session, model_path = load_edge_runtime()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    runtime_stats = measure_edge_runtime(str(model_path))
    render_toolbar(signal_bank)

    signal, event_timestamp, signal_kind = get_selected_signal(signal_bank)
    signal_token = f"{signal_kind}:{st.session_state.sample_number}"
    if st.session_state.active_signal_token != signal_token:
        st.session_state.playback_cursor = 0
        st.session_state.active_signal_token = signal_token

    analysis = build_signal_analysis(signal, event_timestamp, session)
    alert_scores, detections = get_score_bundle(
        analysis,
        st.session_state.alert_engine,
        st.session_state.alert_threshold,
    )

    advance_playback(analysis["sample_length"])
    cursor = min(st.session_state.playback_cursor, analysis["sample_length"] - 1)
    start_idx, end_idx = get_visible_range(analysis["sample_length"], cursor)

    model_score = get_current_window_score(analysis["centers"], analysis["model_scores"], cursor)
    impact_score = get_current_window_score(analysis["centers"], analysis["impact_scores"], cursor)
    alert_score = get_current_window_score(analysis["centers"], alert_scores, cursor)
    active_alert = get_active_alert(detections, cursor)
    detection_log = format_log_entries(detections, cursor, event_timestamp)

    render_hero(
        st.session_state.alert_engine,
        st.session_state.signal_family,
        runtime_stats,
        cursor / SAMPLING_RATE_HZ,
        alert_score,
    )

    content_left, content_right = st.columns([1.65, 0.95], gap="large")

    with content_left:
        st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Live vibration feed</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-copy">The amber cursor advances through the selected 10-second stream while the chart scrolls over the latest 5 seconds.</div>',
            unsafe_allow_html=True,
        )
        signal_fig = create_signal_chart(
            analysis["signal"],
            start_idx,
            end_idx,
            cursor,
            detections,
            event_timestamp,
        )
        st.plotly_chart(signal_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div style="height: 0.9rem;"></div>', unsafe_allow_html=True)

        st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Detection signals</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-copy">Amber drives the live alert banner. Red shows the adaptive impact signal. Blue shows the exported ONNX model probability.</div>',
            unsafe_allow_html=True,
        )
        score_fig = create_score_chart(
            analysis,
            start_idx,
            end_idx,
            cursor,
            alert_scores,
            st.session_state.alert_threshold,
            event_timestamp,
        )
        st.plotly_chart(score_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with content_right:
        st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
        render_status_and_metrics(
            signal_kind,
            cursor,
            event_timestamp,
            model_score,
            impact_score,
            alert_score,
            detections,
            active_alert,
        )
        render_detection_log(detection_log)
        st.markdown(
            '<div class="footer-note">Try switching between an event stream and a nominal stream, then change the alert engine to compare the live behavior.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    render_sidebar(
        runtime_stats,
        st.session_state.signal_family,
        st.session_state.alert_engine,
        event_timestamp,
    )

    if st.session_state.playback_running:
        time.sleep(REFRESH_INTERVAL_SEC)
        st.rerun()


if __name__ == "__main__":
    main()
