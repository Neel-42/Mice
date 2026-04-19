"""Post-process binary seizure probabilities to reduce false positives."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .features import featurize_recording
from .load_data import load_recording


def smooth_prob(p: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return p.copy()
    k = int(k)
    ker = np.ones(k, dtype=float) / float(k)
    return np.convolve(p, ker, mode="same")


def spans_from_binary(y: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    in_span = False
    s = 0
    for i, v in enumerate(y.astype(int)):
        if v == 1 and not in_span:
            in_span = True
            s = i
        elif v == 0 and in_span:
            spans.append((s, i - 1))
            in_span = False
    if in_span:
        spans.append((s, len(y) - 1))
    return spans


def binary_from_spans(n: int, spans: list[tuple[int, int]]) -> np.ndarray:
    y = np.zeros(n, dtype=np.int32)
    for s, e in spans:
        y[s : e + 1] = 1
    return y


def merge_close_spans(spans: list[tuple[int, int]], max_gap_windows: int) -> list[tuple[int, int]]:
    if not spans:
        return []
    out = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = out[-1]
        if s - pe - 1 <= max_gap_windows:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def filter_min_duration(spans: list[tuple[int, int]], min_windows: int) -> list[tuple[int, int]]:
    return [(s, e) for (s, e) in spans if (e - s + 1) >= min_windows]


def apply_postprocess(
    p1: np.ndarray,
    threshold: float,
    smooth_k: int,
    min_duration_windows: int,
    merge_gap_windows: int,
) -> np.ndarray:
    ps = smooth_prob(p1, smooth_k)
    y = (ps >= threshold).astype(np.int32)
    spans = spans_from_binary(y)
    spans = merge_close_spans(spans, merge_gap_windows)
    spans = filter_min_duration(spans, min_duration_windows)
    return binary_from_spans(len(y), spans)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("models/seizure_binary_rf.joblib"))
    ap.add_argument("--mat", nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--base-threshold", type=float, default=0.02)
    ap.add_argument("--target-recall", type=float, default=0.75)
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    clf = bundle["model"]
    win_s = float(bundle["win_s"])
    stride_s = float(bundle["stride_s"])
    label_mode = str(bundle["label_mode"])
    classes = list(clf.classes_)
    if 1 not in classes:
        raise ValueError("Expected binary model with class 1.")
    pos_idx = classes.index(1)

    per_file = []
    all_y = []
    all_p = []
    for mat in args.mat:
        rec = load_recording(mat)
        X, y_agg = featurize_recording(
            rec.ecog, rec.swd_label, rec.fs_hz, win_s=win_s, stride_s=stride_s, label_mode=label_mode
        )
        y_true = (y_agg > 0).astype(np.int32)
        p1 = clf.predict_proba(X)[:, pos_idx]
        per_file.append((mat, y_true, p1))
        all_y.append(y_true)
        all_p.append(p1)

    y_all = np.concatenate(all_y)
    p_all = np.concatenate(all_p)

    grid = []
    for smooth_k in [1, 3, 5, 7, 11]:
        for min_dur_s in [2, 3, 4, 5, 6, 8, 10]:
            for merge_gap_s in [0, 1, 2, 3, 4]:
                for thr in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
                    min_w = max(1, int(round(min_dur_s / stride_s)))
                    gap_w = max(0, int(round(merge_gap_s / stride_s)))
                    yp = apply_postprocess(
                        p_all,
                        threshold=thr,
                        smooth_k=smooth_k,
                        min_duration_windows=min_w,
                        merge_gap_windows=gap_w,
                    )
                    m = metrics(y_all, yp)
                    m.update(
                        {
                            "threshold": thr,
                            "smooth_k": smooth_k,
                            "min_duration_s": min_dur_s,
                            "merge_gap_s": merge_gap_s,
                        }
                    )
                    grid.append(m)
    grid_df = pd.DataFrame(grid)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(args.out_dir / "postprocess_grid_search.csv", index=False)

    # choose highest precision among configs that satisfy target recall
    ok = grid_df[grid_df["recall"] >= args.target_recall]
    if ok.empty:
        best = grid_df.sort_values(["f1", "precision"], ascending=False).iloc[0]
    else:
        best = ok.sort_values(["precision", "f1"], ascending=False).iloc[0]

    best_cfg = {
        "threshold": float(best["threshold"]),
        "smooth_k": int(best["smooth_k"]),
        "min_duration_s": float(best["min_duration_s"]),
        "merge_gap_s": float(best["merge_gap_s"]),
    }

    # write per-file postprocessed predictions
    metrics_rows = []
    for mat, y_true, p1 in per_file:
        min_w = max(1, int(round(best_cfg["min_duration_s"] / stride_s)))
        gap_w = max(0, int(round(best_cfg["merge_gap_s"] / stride_s)))
        y_post = apply_postprocess(
            p1,
            threshold=best_cfg["threshold"],
            smooth_k=best_cfg["smooth_k"],
            min_duration_windows=min_w,
            merge_gap_windows=gap_w,
        )
        m = metrics(y_true, y_post)
        m["file"] = Path(mat).name
        m["n_windows"] = int(len(y_true))
        metrics_rows.append(m)

        starts = np.arange(len(y_true), dtype=float) * stride_s
        out_csv = args.out_dir / f"predictions_postprocessed_{Path(mat).stem}.csv"
        pd.DataFrame(
            {
                "window_start_s": starts,
                "label_true": y_true,
                "p_class_1": p1,
                "label_pred_post": y_post,
            }
        ).to_csv(out_csv, index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(args.out_dir / "postprocess_metrics_per_file.csv", index=False)
    pd.DataFrame([best_cfg]).to_csv(args.out_dir / "postprocess_best_config.csv", index=False)

    print("Best postprocess config:", best_cfg)
    print(metrics_df.to_string(index=False))
    print("Saved:")
    print("-", (args.out_dir / "postprocess_grid_search.csv").resolve())
    print("-", (args.out_dir / "postprocess_best_config.csv").resolve())
    print("-", (args.out_dir / "postprocess_metrics_per_file.csv").resolve())


if __name__ == "__main__":
    main()
