"""Create plots showing which engineered features influence seizure predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .features import featurize_recording
from .load_data import load_recording


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("models/seizure_binary_rf.joblib"))
    p.add_argument("--mat", nargs="+", required=True)
    p.add_argument("--threshold", type=float, default=0.02)
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument(
        "--max-windows-for-permutation",
        type=int,
        default=25000,
        help="Cap windows used for permutation importance (for speed).",
    )
    args = p.parse_args()

    bundle = joblib.load(args.model)
    clf = bundle["model"]
    feature_names = list(bundle["feature_names"])
    win_s = float(bundle["win_s"])
    stride_s = float(bundle["stride_s"])
    label_mode = str(bundle["label_mode"])
    classes = list(clf.classes_)
    if 1 not in classes:
        raise ValueError("Binary seizure class (1) is missing in model classes.")
    pos_idx = classes.index(1)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
    for mat in args.mat:
        rec = load_recording(mat)
        X, y_agg = featurize_recording(
            rec.ecog,
            rec.swd_label,
            rec.fs_hz,
            win_s=win_s,
            stride_s=stride_s,
            label_mode=label_mode,
        )
        y = (y_agg > 0).astype(np.int32)
        p1 = clf.predict_proba(X)[:, pos_idx]
        X_parts.append(X)
        y_parts.append(y)
        p_parts.append(p1)

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)
    p_all = np.concatenate(p_parts)
    y_hat = (p_all >= args.threshold).astype(np.int32)
    bacc = balanced_accuracy_score(y_all, y_hat)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Model's internal feature importance (tree-based)
    fi = pd.DataFrame(
        {"feature": feature_names, "importance_gini": clf.feature_importances_}
    ).sort_values("importance_gini", ascending=False)
    fi.to_csv(args.out_dir / "feature_importance_gini.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(fi["feature"][::-1], fi["importance_gini"][::-1], color="steelblue")
    plt.xlabel("Relative importance (RandomForest Gini)")
    plt.ylabel("Feature")
    plt.title("Feature importance ranking")
    plt.tight_layout()
    plt.savefig(args.out_dir / "feature_importance_gini.png", dpi=160)
    plt.close()

    # 2) Permutation importance on a sampled subset (balanced accuracy scorer)
    n = X_all.shape[0]
    if n > args.max_windows_for_permutation:
        rng = np.random.default_rng(42)
        sel = rng.choice(n, size=args.max_windows_for_permutation, replace=False)
        Xp = X_all[sel]
        yp = y_all[sel]
    else:
        Xp = X_all
        yp = y_all

    perm = permutation_importance(
        clf,
        Xp,
        yp,
        n_repeats=8,
        random_state=42,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    pfi = pd.DataFrame(
        {
            "feature": feature_names,
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_std": perm.importances_std,
        }
    ).sort_values("perm_importance_mean", ascending=False)
    pfi.to_csv(args.out_dir / "feature_importance_permutation.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.barh(
        pfi["feature"][::-1],
        pfi["perm_importance_mean"][::-1],
        xerr=pfi["perm_importance_std"][::-1],
        color="darkorange",
        alpha=0.9,
    )
    plt.xlabel("Drop in balanced accuracy when permuted")
    plt.ylabel("Feature")
    plt.title("Permutation importance (more robust influence estimate)")
    plt.tight_layout()
    plt.savefig(args.out_dir / "feature_importance_permutation.png", dpi=160)
    plt.close()

    # 3) Influence trend plots: each feature vs predicted seizure probability
    # Bin each feature into quantiles and show mean p(seizure) in each bin.
    trend_rows = []
    for j, f in enumerate(feature_names):
        x = X_all[:, j].astype(float)
        q_edges = np.quantile(x, np.linspace(0, 1, 11))
        q_edges = np.unique(q_edges)
        if q_edges.shape[0] < 4:
            continue
        bins = pd.cut(x, bins=q_edges, include_lowest=True, duplicates="drop")
        g = pd.DataFrame({"bin": bins, "p1": p_all}).groupby("bin", observed=True).agg(
            p1_mean=("p1", "mean"),
            p1_std=("p1", "std"),
            n=("p1", "count"),
        )
        centers = []
        for interval in g.index:
            centers.append((float(interval.left) + float(interval.right)) / 2.0)
        g = g.reset_index(drop=True)
        g["feature"] = f
        g["x_center"] = centers
        trend_rows.append(g[["feature", "x_center", "p1_mean", "p1_std", "n"]])

    if trend_rows:
        trend_df = pd.concat(trend_rows, ignore_index=True)
        trend_df.to_csv(args.out_dir / "feature_probability_trends.csv", index=False)

        top_feats = list(pfi["feature"].head(6))
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        axes = axes.ravel()
        for i, feat in enumerate(top_feats):
            ax = axes[i]
            t = trend_df[trend_df["feature"] == feat]
            ax.plot(t["x_center"], t["p1_mean"], marker="o", linewidth=1.4)
            ax.fill_between(
                t["x_center"],
                np.clip(t["p1_mean"] - t["p1_std"], 0, 1),
                np.clip(t["p1_mean"] + t["p1_std"], 0, 1),
                alpha=0.2,
            )
            ax.set_title(feat)
            ax.set_xlabel("Feature value (binned center)")
            ax.set_ylabel("Mean predicted p(seizure)")
            ax.set_ylim(0, 1)
        for k in range(len(top_feats), 6):
            axes[k].axis("off")
        plt.suptitle("How top features shift seizure probability", y=1.02)
        plt.savefig(args.out_dir / "feature_influence_trends_top6.png", dpi=160)
        plt.close()

    print(f"Pooled balanced accuracy at threshold {args.threshold:.3f}: {bacc:.4f}")
    print("Wrote:")
    print(" -", (args.out_dir / "feature_importance_gini.csv").resolve())
    print(" -", (args.out_dir / "feature_importance_gini.png").resolve())
    print(" -", (args.out_dir / "feature_importance_permutation.csv").resolve())
    print(" -", (args.out_dir / "feature_importance_permutation.png").resolve())
    print(" -", (args.out_dir / "feature_probability_trends.csv").resolve())
    print(" -", (args.out_dir / "feature_influence_trends_top6.png").resolve())


if __name__ == "__main__":
    main()
