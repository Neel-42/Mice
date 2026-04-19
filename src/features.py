"""Per-window spectral and time-domain features for rodent ECoG."""

from __future__ import annotations

import numpy as np


def _bandpower_fft(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    x = x - np.mean(x)
    n = x.shape[0]
    spec = np.fft.rfft(x * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= f_lo) & (freqs < f_hi)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(spec[mask]) ** 2))


def extract_window_features(window: np.ndarray, fs: float) -> np.ndarray:
    """Return a 1D feature vector for one window of ECoG."""
    w = window.astype(np.float64)
    line_length = float(np.sum(np.abs(np.diff(w))))
    rms = float(np.sqrt(np.mean(w**2)))
    activity = float(np.mean(np.abs(w)))
    var = float(np.var(w))
    zc = int(np.sum(np.diff(np.signbit(w - np.mean(w))) != 0))

    # Bands chosen for rodent SWD / cortical slow oscillations vs faster activity
    bands = (
        (1.0, 4.0),
        (4.0, 8.0),
        (8.0, 13.0),
        (13.0, 30.0),
        (30.0, min(90.0, fs / 2 - 1.0)),
    )
    bp = [_bandpower_fft(w, fs, lo, hi) for lo, hi in bands]
    feats = [rms, activity, var, line_length, float(zc)] + bp
    return np.array(feats, dtype=np.float32)


def featurize_recording(
    ecog: np.ndarray,
    labels: np.ndarray,
    fs: float,
    win_s: float = 2.0,
    stride_s: float = 1.0,
    label_mode: str = "mode",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over the recording and aggregate labels per window.

    label_mode:
      - 'mode': most common integer label in window (rounded)
      - 'max': max label in window (flags any SWD overlap)
    """
    if ecog.shape != labels.shape:
        raise ValueError("ecog and labels must have same length")
    win = int(round(win_s * fs))
    step = int(round(stride_s * fs))
    if win < 16 or step < 1:
        raise ValueError("Bad window / stride for sampling rate")

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for start in range(0, len(ecog) - win + 1, step):
        seg = ecog[start : start + win]
        lab_seg = labels[start : start + win]
        if label_mode == "max":
            y = int(np.round(np.max(lab_seg)))
        else:
            ri = np.rint(lab_seg).astype(int)
            vals, cnts = np.unique(ri, return_counts=True)
            y = int(vals[np.argmax(cnts)])
        xs.append(extract_window_features(seg, fs))
        ys.append(y)
    return np.stack(xs, axis=0), np.array(ys, dtype=np.int32)
