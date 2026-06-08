"""Data helpers for the interactive seizure ECoG viewer."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .load_data import Recording, load_recording

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RECORDINGS: dict[str, dict[str, str]] = {
    "rec1": {
        "label": "Mouse 30 — Recording 1",
        "mat": "m30_Rec1_ALL (1).mat",
        "postprocessed_csv": "outputs/predictions_postprocessed_m30_Rec1_ALL (1).csv",
    },
    "rec2": {
        "label": "Mouse 30 — Recording 2",
        "mat": "m30_Rec2_ALL (1).mat",
        "postprocessed_csv": "outputs/predictions_postprocessed_m30_Rec2_ALL (1).csv",
    },
}

# Featured predicted-seizure zoom segments (seconds) from earlier analysis.
FEATURED_SEGMENTS: dict[str, list[dict[str, float | str]]] = {
    "rec1": [
        {"id": 1, "start_s": 11025.0, "end_s": 11071.0, "label": "Predicted seizure #1"},
        {"id": 2, "start_s": 11451.0, "end_s": 11497.0, "label": "Predicted seizure #2"},
        {"id": 3, "start_s": 11781.0, "end_s": 11827.0, "label": "Predicted seizure #3"},
        {"id": 4, "start_s": 11823.0, "end_s": 11871.0, "label": "Predicted seizure #4"},
        {"id": 5, "start_s": 12211.0, "end_s": 12256.0, "label": "Predicted seizure #5"},
    ],
}


@dataclass
class Span:
    start_s: float
    end_s: float

    def to_dict(self) -> dict[str, float]:
        return {"start_s": self.start_s, "end_s": self.end_s}


@lru_cache(maxsize=4)
def get_recording(rec_id: str) -> Recording:
    if rec_id not in RECORDINGS:
        raise KeyError(f"Unknown recording id: {rec_id}")
    mat = PROJECT_ROOT / RECORDINGS[rec_id]["mat"]
    return load_recording(mat)


def spans_from_sample_labels(labels: np.ndarray, fs: float) -> list[Span]:
    spans: list[Span] = []
    in_span = False
    s = 0
    for i, v in enumerate(labels):
        active = float(v) > 0.5
        if active and not in_span:
            in_span = True
            s = i
        elif not active and in_span:
            in_span = False
            spans.append(Span(s / fs, i / fs))
    if in_span:
        spans.append(Span(s / fs, len(labels) / fs))
    return spans


def spans_from_window_labels(
    y: np.ndarray, starts_s: np.ndarray, win_s: float
) -> list[Span]:
    spans: list[Span] = []
    in_span = False
    s_idx = 0
    stride_s = float(starts_s[1] - starts_s[0]) if len(starts_s) > 1 else 1.0
    for i, v in enumerate(y.astype(int)):
        if v == 1 and not in_span:
            in_span = True
            s_idx = i
        elif v == 0 and in_span:
            in_span = False
            spans.append(
                Span(float(starts_s[s_idx]), float(starts_s[i - 1]) + win_s)
            )
    if in_span:
        spans.append(
            Span(float(starts_s[s_idx]), float(starts_s[-1]) + win_s)
        )
    _ = stride_s
    return spans


def get_true_spans(rec_id: str) -> list[dict[str, float]]:
    rec = get_recording(rec_id)
    return [s.to_dict() for s in spans_from_sample_labels(rec.swd_label, rec.fs_hz)]


def get_predicted_spans(rec_id: str, win_s: float = 2.0) -> list[dict[str, float]]:
    csv_path = PROJECT_ROOT / RECORDINGS[rec_id]["postprocessed_csv"]
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    starts = df["window_start_s"].to_numpy()
    y = df["label_pred_post"].to_numpy()
    return [s.to_dict() for s in spans_from_window_labels(y, starts, win_s)]


def downsample_trace(
    ecog: np.ndarray, fs: float, t0: float, t1: float, max_points: int = 12000
) -> dict[str, list[float]]:
    t0 = max(0.0, t0)
    t1 = min(len(ecog) / fs, t1)
    if t1 <= t0:
        return {"t": [], "ecog": []}
    s0 = int(t0 * fs)
    s1 = int(t1 * fs)
    seg = ecog[s0:s1]
    n = seg.shape[0]
    if n == 0:
        return {"t": [], "ecog": []}
    step = max(1, int(np.ceil(n / max_points)))
    idx = np.arange(0, n, step)
    t = (s0 + idx) / fs
    return {"t": t.astype(float).tolist(), "ecog": seg[idx].astype(float).tolist()}


def spans_in_window(spans: list[dict[str, float]], t0: float, t1: float) -> list[dict[str, float]]:
    out = []
    for sp in spans:
        if sp["end_s"] < t0 or sp["start_s"] > t1:
            continue
        out.append(
            {
                "start_s": max(t0, sp["start_s"]),
                "end_s": min(t1, sp["end_s"]),
            }
        )
    return out


def recording_meta(rec_id: str) -> dict[str, Any]:
    rec = get_recording(rec_id)
    return {
        "id": rec_id,
        "label": RECORDINGS[rec_id]["label"],
        "mouse_id": rec.mouse_id,
        "fs_hz": rec.fs_hz,
        "duration_s": rec.duration_s,
        "duration_h": rec.duration_s / 3600.0,
        "n_true_spans": len(get_true_spans(rec_id)),
        "n_pred_spans": len(get_predicted_spans(rec_id)),
    }
