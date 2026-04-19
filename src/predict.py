"""Run trained model on a recording and export per-window predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import featurize_recording
from .load_data import load_recording


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mat", required=True, help="Recording .mat path")
    p.add_argument("--model", type=Path, default=Path("models/seizure_rf.joblib"))
    p.add_argument("--out-csv", type=Path, default=Path("outputs/predictions.csv"))
    args = p.parse_args()

    bundle = joblib.load(args.model)
    clf = bundle["model"]
    win_s = float(bundle["win_s"])
    stride_s = float(bundle["stride_s"])
    label_mode = str(bundle["label_mode"])
    task = str(bundle.get("task", "multiclass"))

    rec = load_recording(args.mat)
    X, y_agg = featurize_recording(
        rec.ecog,
        rec.swd_label,
        rec.fs_hz,
        win_s=win_s,
        stride_s=stride_s,
        label_mode=label_mode,
    )
    if task == "binary":
        y_true = (y_agg > 0).astype(np.int32)
    else:
        y_true = y_agg.astype(np.int32)
    y_pred = clf.predict(X)
    proba = clf.predict_proba(X)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    t0 = (np.arange(X.shape[0]) * stride_s).astype(np.float32)
    df = pd.DataFrame(
        {
            "window_start_s": t0,
            "label_true": y_true,
            "label_pred": y_pred.astype(np.int32),
        }
    )
    if task == "binary":
        df["swdlabel_max_in_window"] = y_agg.astype(np.int32)
    classes = [int(c) for c in clf.classes_]
    for j, c in enumerate(classes):
        df[f"p_class_{c}"] = proba[:, j]
    df.to_csv(args.out_csv, index=False)
    acc = float((y_pred == y_true).mean())
    print(f"Window accuracy ({task}): {acc:.4f}")
    print("Wrote", args.out_csv.resolve())


if __name__ == "__main__":
    main()
