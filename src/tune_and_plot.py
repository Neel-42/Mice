"""Tune binary seizure threshold and generate EEG visualization plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

from .features import featurize_recording
from .load_data import load_recording


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    bacc = balanced_accuracy_score(y_true, y_pred)
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "bacc": float(bacc)}


def spans_from_binary(y: np.ndarray, stride_s: float, win_s: float) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    in_span = False
    start_idx = 0
    for i, val in enumerate(y.astype(int)):
        if val == 1 and not in_span:
            in_span = True
            start_idx = i
        elif val == 0 and in_span:
            in_span = False
            s = start_idx * stride_s
            e = (i - 1) * stride_s + win_s
            spans.append((s, e))
    if in_span:
        s = start_idx * stride_s
        e = (len(y) - 1) * stride_s + win_s
        spans.append((s, e))
    return spans


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("models/seizure_binary_rf.joblib"))
    p.add_argument("--mat", nargs="+", required=True)
    p.add_argument("--target-recall", type=float, default=0.92)
    p.add_argument("--plot-mat", default=None, help="Which recording to visualize")
    p.add_argument("--plot-start-s", type=float, default=0.0)
    p.add_argument("--plot-duration-s", type=float, default=600.0)
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = p.parse_args()

    bundle = joblib.load(args.model)
    clf = bundle["model"]
    win_s = float(bundle["win_s"])
    stride_s = float(bundle["stride_s"])
    label_mode = str(bundle["label_mode"])

    all_probs = []
    all_true = []
    by_file: list[tuple[str, np.ndarray, np.ndarray]] = []
    for mat in args.mat:
        rec = load_recording(mat)
        X, y_agg = featurize_recording(
            rec.ecog, rec.swd_label, rec.fs_hz, win_s=win_s, stride_s=stride_s, label_mode=label_mode
        )
        y_true = (y_agg > 0).astype(np.int32)
        classes = list(clf.classes_)
        if 1 in classes:
            p1 = clf.predict_proba(X)[:, classes.index(1)]
        else:
            p1 = np.zeros_like(y_true, dtype=float)
        all_probs.append(p1)
        all_true.append(y_true)
        by_file.append((mat, y_true, p1))

    y_true_all = np.concatenate(all_true)
    p_all = np.concatenate(all_probs)

    thresholds = np.linspace(0.02, 0.95, 120)
    rows = []
    for t in thresholds:
        y_pred = (p_all >= t).astype(np.int32)
        m = compute_metrics(y_true_all, y_pred)
        m["threshold"] = float(t)
        rows.append(m)
    df = pd.DataFrame(rows)

    meets = df[df["recall"] >= args.target_recall].copy()
    if not meets.empty:
        best = meets.sort_values(["f1", "bacc"], ascending=False).iloc[0]
    else:
        best = df.sort_values(["recall", "f1"], ascending=False).iloc[0]
    best_t = float(best["threshold"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "binary_threshold_sweep.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(df["threshold"], df["recall"], label="Recall (ictal)")
    plt.plot(df["threshold"], df["precision"], label="Precision (ictal)")
    plt.plot(df["threshold"], df["f1"], label="F1")
    plt.plot(df["threshold"], df["bacc"], label="Balanced accuracy")
    plt.axvline(best_t, color="k", linestyle="--", label=f"chosen t={best_t:.3f}")
    plt.xlabel("Probability threshold for ictal class")
    plt.ylabel("Metric value")
    plt.title("Binary threshold tuning")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(args.out_dir / "binary_threshold_metrics.png", dpi=150)
    plt.close()

    per_file_rows = []
    for mat, y_true, p1 in by_file:
        y_pred = (p1 >= best_t).astype(np.int32)
        m = compute_metrics(y_true, y_pred)
        m["file"] = Path(mat).name
        per_file_rows.append(m)
    pd.DataFrame(per_file_rows).to_csv(args.out_dir / "binary_metrics_per_file.csv", index=False)

    plot_target = args.plot_mat if args.plot_mat else args.mat[0]
    recp = load_recording(plot_target)
    Xp, y_aggp = featurize_recording(
        recp.ecog,
        recp.swd_label,
        recp.fs_hz,
        win_s=win_s,
        stride_s=stride_s,
        label_mode=label_mode,
    )
    y_true_plot = (y_aggp > 0).astype(np.int32)
    p1_plot = clf.predict_proba(Xp)[:, list(clf.classes_).index(1)]
    y_pred_plot = (p1_plot >= best_t).astype(np.int32)

    t0 = max(0.0, args.plot_start_s)
    t1 = min(recp.duration_s, t0 + args.plot_duration_s)
    s0 = int(t0 * recp.fs_hz)
    s1 = int(t1 * recp.fs_hz)
    ecog = recp.ecog[s0:s1]
    t = np.arange(ecog.shape[0]) / recp.fs_hz + t0

    true_spans = spans_from_binary(y_true_plot, stride_s, win_s)
    pred_spans = spans_from_binary(y_pred_plot, stride_s, win_s)

    plt.figure(figsize=(13, 4))
    plt.plot(t, ecog, linewidth=0.6, color="navy", alpha=0.9, label="ECoG")
    for s, e in true_spans:
        if e < t0 or s > t1:
            continue
        plt.axvspan(max(s, t0), min(e, t1), color="green", alpha=0.10)
    for s, e in pred_spans:
        if e < t0 or s > t1:
            continue
        plt.axvspan(max(s, t0), min(e, t1), color="red", alpha=0.08)
    plt.xlabel("Time (s)")
    plt.ylabel("ECoG level")
    plt.title(
        f"ECoG trace with seizure windows (green=true, red=pred), threshold={best_t:.3f}"
    )
    plt.tight_layout()
    plt.savefig(args.out_dir / "ecog_time_vs_level.png", dpi=150)
    plt.close()

    pooled_pred = (p_all >= best_t).astype(np.int32)
    pooled = compute_metrics(y_true_all, pooled_pred)
    print(f"Chosen threshold: {best_t:.4f}")
    print(
        "Pooled binary metrics:",
        f"precision={pooled['precision']:.4f}",
        f"recall={pooled['recall']:.4f}",
        f"f1={pooled['f1']:.4f}",
        f"balanced_acc={pooled['bacc']:.4f}",
    )
    print("Saved:")
    print(" -", (args.out_dir / "binary_threshold_sweep.csv").resolve())
    print(" -", (args.out_dir / "binary_threshold_metrics.png").resolve())
    print(" -", (args.out_dir / "binary_metrics_per_file.csv").resolve())
    print(" -", (args.out_dir / "ecog_time_vs_level.png").resolve())


if __name__ == "__main__":
    main()
