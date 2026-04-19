"""Join literature-based gene→seizure phenotype tables with optional per-mouse metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def load_gene_table(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def load_mouse_metadata(json_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def genes_for_mouse(mouse_id: int, meta: dict[str, Any]) -> list[str]:
    entry = meta.get(str(mouse_id))
    if not entry:
        return []
    g = entry.get("genotype", "")
    parts: list[str] = []
    for token in g.replace(";", " ").split():
        token = token.strip()
        if not token:
            continue
        m = re.match(r"([A-Za-z][A-Za-z0-9]*)", token)
        if m:
            parts.append(m.group(1))
    return parts


def interpret_mouse_genes(
    mouse_id: int,
    gene_csv: str | Path,
    metadata_json: str | Path | None = None,
) -> pd.DataFrame:
    """
    Return rows from the gene association table that plausibly match genotypes
    listed for this mouse_id in metadata (substring match on gene symbol).
    """
    table = load_gene_table(gene_csv)
    if metadata_json is None:
        return table.iloc[0:0]

    meta = load_mouse_metadata(metadata_json)
    tokens = genes_for_mouse(mouse_id, meta)
    if not tokens:
        return table.iloc[0:0]

    hits = []
    for tok in tokens:
        sym = tok.replace("+/-", "").replace("-/-", "").replace("+/+", "")
        sym = sym.strip()
        if not sym:
            continue
        m = table[table["gene_symbol"].str.contains(sym, case=False, regex=False)]
        hits.append(m)
    if not hits:
        return table.iloc[0:0]
    return pd.concat(hits, ignore_index=True).drop_duplicates()


def label_code_to_phenotype(code: int) -> str:
    """
    Heuristic mapping for this dataset's SWDlabel integers.
    Your lab's coding may differ — edit this function to match your key.
    """
    mapping = {
        0: "interictal / baseline",
        1: "SWD subtype A (update name from lab key)",
        3: "SWD subtype B",
        5: "SWD subtype C",
        6: "SWD subtype D",
    }
    return mapping.get(int(code), f"unknown code {code}")
