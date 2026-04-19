"""Summarize labels in a recording and optional genotype→phenotype hints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .gene_phenotype import interpret_mouse_genes, label_code_to_phenotype
from .load_data import load_recording


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mat", required=True)
    p.add_argument(
        "--gene-csv",
        type=Path,
        default=Path("data/gene_seizure_associations.csv"),
    )
    p.add_argument("--metadata-json", type=Path, default=None)
    args = p.parse_args()

    rec = load_recording(args.mat)
    print("File:", rec.source_path)
    print("Mouse ID:", rec.mouse_id, "fs:", rec.fs_hz, "duration_h:", rec.duration_s / 3600)
    y = rec.swd_label
    vals, cnts = np.unique(np.rint(y).astype(int), return_counts=True)
    print("Label codes (SWDlabel) counts:")
    for v, c in zip(vals, cnts):
        print(f"  {int(v):d}: {int(c):d}  ({label_code_to_phenotype(int(v))})")

    if args.metadata_json and args.metadata_json.exists():
        df = interpret_mouse_genes(rec.mouse_id, args.gene_csv, args.metadata_json)
        if df.empty:
            print("No genotype match in metadata for gene table lookup.")
        else:
            print("\nLiterature-based genotype associations (curated starter table):")
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
