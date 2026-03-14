from __future__ import annotations

import numpy as np


def t1_magnitude_model(t_ms: np.ndarray, offset_mV: float, amplitude_mV: float, t1_ms: float) -> np.ndarray:
    """Magnitude-detector model for inversion recovery."""
    return offset_mV + amplitude_mV * np.abs(1.0 - 2.0 * np.exp(-t_ms / t1_ms))


def t2_envelope_model(t_ms: np.ndarray, amplitude_mV: float, t2_ms: float) -> np.ndarray:
    """Single-exponential echo-envelope model."""
    return amplitude_mV * np.exp(-t_ms / t2_ms)
