"""Train a multiclass model on windowed ECoG features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from .features import featurize_recording
from .load_data import load_recording, time_train_test_split


def oversample_training_set(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    minority_frac_of_majority: float = 0.05,
    min_target: int = 1500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample rare window labels so the forest sees enough positives.
    Majority class (usually 0) is left unchanged.
    """
    classes, counts = np.unique(y, return_counts=True)
    order = np.argsort(-counts)
    classes = classes[order]
    counts = counts[order]
    majority_n = int(counts[0])
    target = max(min_target, int(minority_frac_of_majority * majority_n))

    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    for c, n in zip(classes, counts):
        idx = np.flatnonzero(y == c)
        if int(n) >= target:
            chunks_x.append(X[idx])
            chunks_y.append(y[idx])
            continue
        pick = rng.choice(idx, size=target, replace=True)
        chunks_x.append(X[pick])
        chunks_y.append(y[pick])
    Xb = np.vstack(chunks_x)
    yb = np.concatenate(chunks_y)
    perm = rng.permutation(Xb.shape[0])
    return Xb[perm], yb[perm]


def main() -> None:
    p = argparse.ArgumentParser(description="Train SWD / seizure window classifier")
    p.add_argument(
        "--mat",
        nargs="+",
        required=True,
        help="One or more .mat files (v7.3) with rec/ecog and rec/SWDlabel",
    )
    p.add_argument("--win-s", type=float, default=2.0)
    p.add_argument("--stride-s", type=float, default=1.0)
    p.add_argument("--train-frac", type=float, default=0.75)
    p.add_argument("--label-mode", choices=("mode", "max"), default="mode")
    p.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Truncate each recording for faster experiments (uses first N seconds only).",
    )
    p.add_argument("--out", type=Path, default=Path("models/seizure_rf.joblib"))
    p.add_argument(
        "--no-oversample",
        action="store_true",
        help="Disable minority oversampling on the training split.",
    )
    p.add_argument(
        "--binary",
        action="store_true",
        help="Train ictal vs baseline: label=1 if max(SWDlabel)>0 inside the window.",
    )
    args = p.parse_args()

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    meta: dict = {"files": []}

    for m in args.mat:
        rec = load_recording(m)
        ecog = rec.ecog
        lab = rec.swd_label
        if args.max_seconds is not None:
            n = int(rec.fs_hz * args.max_seconds)
            ecog = ecog[:n]
            lab = lab[:n]
        lm = "max" if args.binary else args.label_mode
        X, y = featurize_recording(
            ecog,
            lab,
            rec.fs_hz,
            win_s=args.win_s,
            stride_s=args.stride_s,
            label_mode=lm,
        )
        if args.binary:
            y = (y > 0).astype(np.int32)
        X_parts.append(X)
        y_parts.append(y)
        meta["files"].append(
            {
                "path": rec.source_path,
                "mouse_id": rec.mouse_id,
                "fs_hz": rec.fs_hz,
                "n_windows": int(X.shape[0]),
            }
        )

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)

    n = X_all.shape[0]
    tr_idx, te_idx = time_train_test_split(n, train_frac=args.train_frac)
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_te, y_te = X_all[te_idx], y_all[te_idx]

    rng = np.random.default_rng(42)
    if not args.no_oversample:
        X_tr, y_tr = oversample_training_set(X_tr, y_tr, rng)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    target_names = None
    if args.binary:
        target_names = ["baseline", "ictal"]
    report = classification_report(
        y_te, y_pred, zero_division=0, target_names=target_names
    )
    print(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": clf,
        "feature_names": [
            "rms",
            "mean_abs",
            "var",
            "line_length",
            "zero_crossings",
            "bp_1_4",
            "bp_4_8",
            "bp_8_13",
            "bp_13_30",
            "bp_30_nyq",
        ],
        "win_s": args.win_s,
        "stride_s": args.stride_s,
        "label_mode": "max" if args.binary else args.label_mode,
        "train_frac": args.train_frac,
        "oversampled_train": not args.no_oversample,
        "task": "binary" if args.binary else "multiclass",
    }
    joblib.dump(bundle, args.out)
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved model to", args.out)
    print("Saved meta to", meta_path)


if __name__ == "__main__":
    main()
