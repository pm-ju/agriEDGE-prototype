import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (  # noqa: E402
    CHART_VISIBLE_WINDOW_SEC,
    MODELS_DIR,
    SAMPLING_RATE_HZ,
    STREAM_UPDATE_INTERVAL_MS,
    SYNTHETIC_DATA_DIR,
    WINDOW_SIZE,
    WINDOW_STRIDE,
)


APP_TITLE = "agriEDGE app"
TARGET_SEQUENCE_LENGTH = 500
REFRESH_INTERVAL_SEC = max(0.15, STREAM_UPDATE_INTERVAL_MS / 1000 * 4)

SOURCE_EVENT = "Event sample"
SOURCE_NORMAL = "Normal sample"
MODE_HEURISTIC = "Heuristic score"
MODE_BLEND = "Heuristic + ONNX"


def configure_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_data(show_spinner=False)
def load_signal_bank():
    """Load the saved synthetic signals."""
    return {
        "normal": np.load(SYNTHETIC_DATA_DIR / "normal_samples.npy").astype(np.float32),
        "event": np.load(SYNTHETIC_DATA_DIR / "event_samples.npy").astype(np.float32),
        "event_starts": np.load(SYNTHETIC_DATA_DIR / "event_timestamps.npy").astype(
            np.int64
        ),
    }


@st.cache_resource(show_spinner=False)
def load_edge_runtime():
    """Load the exported ONNX model if it exists."""
    candidate_paths = [
        MODELS_DIR / "detector_quantized.onnx",
        MODELS_DIR / "detector.onnx",
    ]
    model_path = next((path for path in candidate_paths if path.exists()), None)
    if model_path is None:
        return None, None

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    return session, model_path


@st.cache_data(show_spinner=False)
def measure_edge_runtime(model_path_str):
    """Measure rough ONNX runtime once per model file."""
    if model_path_str is None:
        return {
            "model_file": "not loaded",
            "model_size_kb": None,
            "single_window_ms": None,
        }

    model_path = Path(model_path_str)
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(256, 1, TARGET_SEQUENCE_LENGTH).astype(np.float32)

    for _ in range(2):
        session.run(None, {input_name: dummy})

    times = []
    for _ in range(5):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - start) * 1000)

    batch_time = float(np.mean(times))
    return {
        "model_file": model_path.name,
        "model_size_kb": model_path.stat().st_size / 1024,
        "single_window_ms": batch_time / dummy.shape[0],
    }


def initialize_state():
    """Initialize session state."""
    defaults = {
        "signal_source": SOURCE_EVENT,
        "sample_number": 1,
        "score_mode": MODE_BLEND,
        "decision_threshold": 0.68,
        "playback_speed": 1.0,
        "playback_cursor": 0,
        "playback_running": False,
        "loop_playback": True,
        "active_signal_token": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def selected_source_key():
    """Map the UI selection to the stored key."""
    return "event" if st.session_state.signal_source == SOURCE_EVENT else "normal"


def sample_count_for_source(signal_bank):
    """Return the number of saved samples for the current source."""
    return int(signal_bank[selected_source_key()].shape[0])


def clamp_sample_number(signal_bank):
    """Keep the sample number inside the valid range."""
    max_samples = sample_count_for_source(signal_bank)
    if st.session_state.sample_number > max_samples:
        st.session_state.sample_number = max_samples
    if st.session_state.sample_number < 1:
        st.session_state.sample_number = 1


def get_selected_signal(signal_bank):
    """Return the selected signal and event start if present."""
    source_key = selected_source_key()
    sample_index = st.session_state.sample_number - 1

    if source_key == "event":
        signal = signal_bank["event"][sample_index]
        event_start = int(signal_bank["event_starts"][sample_index])
    else:
        signal = signal_bank["normal"][sample_index]
        event_start = None
    return signal, event_start, source_key


def build_windows(signal, window_size, stride):
    """Create overlapping windows."""
    windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)[::stride]
    starts = np.arange(0, len(signal) - window_size + 1, stride)
    centers = starts + window_size // 2
    return windows.astype(np.float32), starts.astype(np.int32), centers.astype(np.int32)


def resample_windows(windows, target_length):
    """Resample each window to the model input length."""
    old_axis = np.linspace(0.0, 1.0, windows.shape[1], dtype=np.float32)
    new_axis = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    return np.stack(
        [np.interp(new_axis, old_axis, row).astype(np.float32) for row in windows],
        axis=0,
    )


def softmax(logits):
    """Compute softmax from ONNX logits."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def compute_model_scores(windows, session):
    """Run ONNX inference and return class-1 probability."""
    if session is None:
        return np.zeros(len(windows), dtype=np.float32)

    input_name = session.get_inputs()[0].name
    resized = resample_windows(windows, TARGET_SEQUENCE_LENGTH)[:, np.newaxis, :]
    logits = session.run(None, {input_name: resized})[0]
    return softmax(logits)[:, 1].astype(np.float32)


def compute_spike_scores(windows):
    """Compute a simple spike score."""
    raw_spike = np.max(np.abs(np.diff(windows, axis=1)), axis=1)
    baseline_count = min(len(raw_spike), max(40, int(1000 / WINDOW_STRIDE)))
    baseline = raw_spike[:baseline_count]
    center = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - center))) + 1e-6
    robust_z = (raw_spike - center) / (1.4826 * mad)
    score = 1.0 / (1.0 + np.exp(-(robust_z - 5.0) / 1.2))
    return score.astype(np.float32)


def select_decision_scores(score_mode, spike_scores, model_scores, model_loaded):
    """Select the score used for thresholding."""
    if score_mode == MODE_BLEND and model_loaded:
        boosted_model = np.clip(model_scores * 3.5, 0.0, 1.0)
        return np.clip(0.82 * spike_scores + 0.18 * boosted_model, 0.0, 1.0)
    return spike_scores


def merge_detections(scores, starts, threshold, window_size):
    """Merge contiguous windows above threshold into regions."""
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


@st.cache_data(show_spinner=False, hash_funcs={ort.InferenceSession: lambda _: None})
def build_signal_analysis_cached(signal_token, signal, _session):
    """Analyze the selected signal once and reuse the result."""
    windows, starts, centers = build_windows(signal, WINDOW_SIZE, WINDOW_STRIDE)
    model_scores = compute_model_scores(windows, _session)
    spike_scores = compute_spike_scores(windows)
    return {
        "signal": signal,
        "starts": starts,
        "centers": centers,
        "model_scores": model_scores,
        "spike_scores": spike_scores,
        "sample_length": len(signal),
    }


def get_score_bundle(analysis, score_mode, threshold, model_loaded):
    """Return the active score and merged detections."""
    active_scores = select_decision_scores(
        score_mode,
        analysis["spike_scores"],
        analysis["model_scores"],
        model_loaded,
    )
    detections = merge_detections(
        active_scores,
        analysis["starts"],
        threshold,
        WINDOW_SIZE,
    )
    return active_scores, detections


def move_cursor(sample_length, step):
    """Move the playback cursor."""
    next_cursor = st.session_state.playback_cursor + step
    if next_cursor < sample_length:
        st.session_state.playback_cursor = next_cursor
        return

    if st.session_state.loop_playback:
        st.session_state.playback_cursor = 0
    else:
        st.session_state.playback_cursor = sample_length - 1
        st.session_state.playback_running = False


def advance_playback(sample_length):
    """Advance playback when auto-play is enabled."""
    if not st.session_state.playback_running:
        return

    step = max(
        WINDOW_STRIDE,
        int(SAMPLING_RATE_HZ * REFRESH_INTERVAL_SEC * st.session_state.playback_speed),
    )
    move_cursor(sample_length, step)


def step_once(sample_length):
    """Move the cursor by one stride."""
    move_cursor(sample_length, WINDOW_STRIDE)


def get_visible_range(sample_length, cursor):
    """Compute the visible range for charts."""
    visible_points = int(CHART_VISIBLE_WINDOW_SEC * SAMPLING_RATE_HZ)
    end_idx = min(sample_length, max(cursor + 1, visible_points))
    start_idx = max(0, end_idx - visible_points)
    return start_idx, end_idx


def get_current_window_score(centers, scores, cursor):
    """Return the score for the current cursor."""
    idx = np.searchsorted(centers, cursor, side="right") - 1
    idx = int(np.clip(idx, 0, len(scores) - 1))
    return float(scores[idx])


def get_active_alert(detections, cursor):
    """Return the current detection if there is one."""
    for detection in detections:
        if detection["start"] <= cursor <= detection["end"]:
            return detection
    return None


def compute_event_offset_ms(detection, event_start):
    """Measure event offset for display."""
    if detection is None or event_start is None:
        return None
    return float(event_start - detection["start"])


def format_event_offset(offset_ms):
    """Format event offset text."""
    if offset_ms is None:
        return "-"
    if offset_ms > 0:
        return f"{offset_ms:.0f} ms before event start"
    if offset_ms < 0:
        return f"{abs(offset_ms):.0f} ms after event start"
    return "at event start"


def create_signal_chart(signal, start_idx, end_idx, cursor, detections, event_start):
    """Create a basic signal chart."""
    visible_signal = signal[start_idx:end_idx]
    x_seconds = np.arange(start_idx, end_idx) / SAMPLING_RATE_HZ

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_seconds,
            y=visible_signal,
            mode="lines",
            name="signal",
        )
    )

    for detection in detections:
        if detection["end"] < start_idx or detection["start"] > end_idx:
            continue
        fig.add_vrect(
            x0=max(detection["start"], start_idx) / SAMPLING_RATE_HZ,
            x1=min(detection["end"], end_idx) / SAMPLING_RATE_HZ,
            fillcolor="rgba(255, 0, 0, 0.15)",
            line_width=0,
        )

    if event_start is not None and start_idx <= event_start <= end_idx:
        fig.add_vline(x=event_start / SAMPLING_RATE_HZ, line_dash="dot")

    fig.add_vline(x=cursor / SAMPLING_RATE_HZ, line_dash="dash")
    fig.update_layout(
        height=320,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
    )
    return fig


def create_score_chart(
    analysis,
    start_idx,
    end_idx,
    cursor,
    alert_scores,
    threshold,
    event_start,
    model_loaded,
):
    """Create a basic score chart."""
    mask = (analysis["centers"] >= start_idx) & (analysis["centers"] <= end_idx)
    x_seconds = analysis["centers"][mask] / SAMPLING_RATE_HZ
    model_scores = analysis["model_scores"][mask]
    alert_slice = alert_scores[mask]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_seconds,
            y=alert_slice,
            mode="lines",
            name="decision score",
        )
    )
    if model_loaded:
        fig.add_trace(
            go.Scatter(
                x=x_seconds,
                y=model_scores,
                mode="lines",
                name="onnx score",
            )
        )

    fig.add_hline(y=threshold, line_dash="dash")
    fig.add_vline(x=cursor / SAMPLING_RATE_HZ, line_dash="dash")
    if event_start is not None and start_idx <= event_start <= end_idx:
        fig.add_vline(x=event_start / SAMPLING_RATE_HZ, line_dash="dot")

    fig.update_layout(
        height=280,
        xaxis_title="Time (s)",
        yaxis_title="Score",
    )
    return fig


def build_detection_rows(detections, cursor, event_start):
    """Build simple rows for display."""
    seen = [d for d in detections if d["start"] <= cursor]
    rows = []
    for detection in seen:
        rows.append(
            {
                "start_s": round(detection["start"] / SAMPLING_RATE_HZ, 3),
                "end_s": round(detection["end"] / SAMPLING_RATE_HZ, 3),
                "score": round(detection["confidence"], 2),
                "event_offset": format_event_offset(
                    compute_event_offset_ms(detection, event_start)
                ),
            }
        )
    return rows


def render_sidebar(signal_bank, runtime_stats):
    """Render simple sidebar controls."""
    with st.sidebar:
        st.header("Controls")
        st.selectbox(
            "Signal source",
            [SOURCE_EVENT, SOURCE_NORMAL],
            key="signal_source",
        )

        clamp_sample_number(signal_bank)
        st.number_input(
            "Sample number",
            min_value=1,
            max_value=sample_count_for_source(signal_bank),
            step=1,
            key="sample_number",
        )
        st.selectbox(
            "Score mode",
            [MODE_BLEND, MODE_HEURISTIC],
            key="score_mode",
        )
        st.slider(
            "Threshold",
            min_value=0.15,
            max_value=0.95,
            step=0.01,
            key="decision_threshold",
        )
        st.slider(
            "Playback speed",
            min_value=0.5,
            max_value=3.0,
            step=0.1,
            key="playback_speed",
        )

        sample_length = int(signal_bank[selected_source_key()].shape[1])
        if st.button("Play / Pause"):
            st.session_state.playback_running = not st.session_state.playback_running
        if st.button("Step once"):
            step_once(sample_length)
        if st.button("Reset"):
            st.session_state.playback_cursor = 0
        st.checkbox("Loop playback", key="loop_playback")

        st.header("Runtime info")
        st.write(f"Model file: {runtime_stats['model_file']}")
        st.write(f"Window size: {WINDOW_SIZE}")
        st.write(f"Stride: {WINDOW_STRIDE}")
        st.write(f"Sampling rate: {SAMPLING_RATE_HZ} Hz")
        if runtime_stats["model_size_kb"] is not None:
            st.write(f"Model size: {runtime_stats['model_size_kb']:.1f} KB")
        if runtime_stats["single_window_ms"] is not None:
            st.write(f"ONNX time per window: {runtime_stats['single_window_ms']:.4f} ms")


def main():
    """Run the Streamlit app."""
    configure_page()
    initialize_state()

    signal_bank = load_signal_bank()
    session, model_path = load_edge_runtime()
    model_loaded = session is not None and model_path is not None
    runtime_stats = measure_edge_runtime(None if model_path is None else str(model_path))

    render_sidebar(signal_bank, runtime_stats)

    signal, event_start, source_key = get_selected_signal(signal_bank)
    signal_token = f"{source_key}:{st.session_state.sample_number}"
    if st.session_state.active_signal_token != signal_token:
        st.session_state.playback_cursor = 0
        st.session_state.active_signal_token = signal_token

    analysis = build_signal_analysis_cached(signal_token, signal, session)
    alert_scores, detections = get_score_bundle(
        analysis,
        st.session_state.score_mode,
        st.session_state.decision_threshold,
        model_loaded,
    )

    advance_playback(analysis["sample_length"])
    cursor = min(st.session_state.playback_cursor, analysis["sample_length"] - 1)
    start_idx, end_idx = get_visible_range(analysis["sample_length"], cursor)
    model_score = get_current_window_score(
        analysis["centers"], analysis["model_scores"], cursor
    )
    alert_score = get_current_window_score(analysis["centers"], alert_scores, cursor)
    active_alert = get_active_alert(detections, cursor)

    st.title(APP_TITLE)
    st.write("Simple viewer for saved vibration samples.")

    if not model_loaded:
        st.warning("ONNX model not found. Showing heuristic score only.")

    st.write(f"Signal type: {source_key}")
    st.write(f"Sample number: {st.session_state.sample_number}")
    st.write(f"Cursor: {cursor / SAMPLING_RATE_HZ:.2f}s")
    st.write(f"Decision score: {alert_score:.2f}")
    if model_loaded:
        st.write(f"ONNX score: {model_score:.2f}")
    if event_start is not None:
        st.write(f"Injected event start: {event_start / SAMPLING_RATE_HZ:.2f}s")

    if active_alert is None:
        st.write("Active detection: none")
    else:
        st.write(
            "Active detection: "
            f"{active_alert['start'] / SAMPLING_RATE_HZ:.3f}s to "
            f"{active_alert['end'] / SAMPLING_RATE_HZ:.3f}s"
        )
        st.write(
            "Event offset: "
            f"{format_event_offset(compute_event_offset_ms(active_alert, event_start))}"
        )

    st.subheader("Signal chart")
    signal_fig = create_signal_chart(
        analysis["signal"],
        start_idx,
        end_idx,
        cursor,
        detections,
        event_start,
    )
    st.plotly_chart(signal_fig, use_container_width=True)

    st.subheader("Score chart")
    score_fig = create_score_chart(
        analysis,
        start_idx,
        end_idx,
        cursor,
        alert_scores,
        st.session_state.decision_threshold,
        event_start,
        model_loaded,
    )
    st.plotly_chart(score_fig, use_container_width=True)

    st.subheader("Detections so far")
    rows = build_detection_rows(detections, cursor, event_start)
    if rows:
        st.write(rows)
    else:
        st.write("No detections yet.")

    if st.session_state.playback_running:
        time.sleep(REFRESH_INTERVAL_SEC)
        st.rerun()


if __name__ == "__main__":
    main()
