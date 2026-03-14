from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ScopeWaveform:
    time_s: np.ndarray
    voltage_v: np.ndarray
    metadata: dict[str, Any]

    @property
    def time_ms(self) -> np.ndarray:
        return self.time_s * 1e3

    @property
    def voltage_mV(self) -> np.ndarray:
        return self.voltage_v * 1e3


def read_scope_csv(path: str | Path) -> ScopeWaveform:
    """Read a GW Instek waveform export such as DS0001A.CSV."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    metadata: dict[str, Any] = {}

    data_start = None
    for i, line in enumerate(lines):
        if line.strip() == "Waveform Data,":
            data_start = i + 1
            break
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            metadata[parts[0]] = parts[1]
    if data_start is None:
        raise ValueError(f"Could not find waveform data block in {path}")

    data = []
    for line in lines[data_start:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            data.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue

    array = np.asarray(data, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Unexpected waveform array shape for {path}: {array.shape}")

    return ScopeWaveform(
        time_s=array[:, 0],
        voltage_v=array[:, 1],
        metadata=metadata,
    )


def load_t1_dataset(path: str | Path) -> pd.DataFrame:
    """Load the hand-transcribed inversion-recovery data table."""
    path = Path(path)
    df = pd.read_csv(path)
    expected = {
        "delay_ms",
        "peak_mV",
        "baseline_mV",
        "amplitude_mV",
        "sigma_amplitude_mV",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def load_zero_crossing_metadata(path: str | Path) -> dict[str, float]:
    """Load zero-crossing metadata used for the primary T1 estimate."""
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "tau0_ms": float(payload["tau0_ms"]),
        "sigma_tau0_ms": float(payload["sigma_tau0_ms"]),
    }
