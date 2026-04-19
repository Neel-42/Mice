"""Load MATLAB v7.3 (.mat) recordings produced by this lab format: top group `rec`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np


@dataclass
class Recording:
    mouse_id: int
    fs_hz: float
    ecog: np.ndarray
    swd_label: np.ndarray
    source_path: str

    @property
    def duration_s(self) -> float:
        return float(self.ecog.shape[0] / self.fs_hz)


def load_recording(mat_path: str | Path) -> Recording:
    mat_path = Path(mat_path)
    with h5py.File(mat_path, "r") as f:
        if "rec" not in f:
            raise ValueError(f"Expected 'rec' group in {mat_path}")
        rec = f["rec"]
        fs = float(np.array(rec["fs"]).squeeze())
        m_id = int(np.array(rec["mID"]).squeeze())
        ecog = np.array(rec["ecog"]).squeeze().astype(np.float32)
        lab = np.array(rec["SWDlabel"]).squeeze()
        if lab.ndim != 1:
            lab = lab.reshape(-1)
        lab = lab.astype(np.float32)
    return Recording(
        mouse_id=m_id,
        fs_hz=fs,
        ecog=ecog,
        swd_label=lab,
        source_path=str(mat_path.resolve()),
    )


def time_train_test_split(
    n_samples: int, train_frac: float = 0.75
) -> Tuple[np.ndarray, np.ndarray]:
    split = int(n_samples * train_frac)
    idx = np.arange(n_samples)
    return idx[:split], idx[split:]
