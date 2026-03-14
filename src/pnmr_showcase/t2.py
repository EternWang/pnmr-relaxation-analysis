from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .io import ScopeWaveform
from .models import t2_envelope_model


@dataclass(slots=True)
class BaselineResult:
    mean_v: float
    rms_v: float
    n_samples: int


@dataclass(slots=True)
class EchoPeaksResult:
    time_ms: np.ndarray
    peak_mV: np.ndarray
    amplitude_mV: np.ndarray
    sigma_amplitude_mV: np.ndarray
    baseline_mV: float
    baseline_rms_mV: float


@dataclass(slots=True)
class T2FitResult:
    amplitude_mV: float
    t2_ms: float
    sigma_amplitude_mV: float
    sigma_t2_ms: float
    covariance: np.ndarray


def estimate_baseline(waveform: ScopeWaveform, t_max_s: float = -0.005) -> BaselineResult:
    """Estimate the pre-trigger baseline mean and RMS."""
    mask = waveform.time_s < t_max_s
    if not np.any(mask):
        raise ValueError("No samples available in the requested baseline window.")
    baseline = float(np.mean(waveform.voltage_v[mask]))
    rms = float(np.std(waveform.voltage_v[mask], ddof=1))
    return BaselineResult(mean_v=baseline, rms_v=rms, n_samples=int(np.sum(mask)))


def extract_echo_peaks(
    waveform: ScopeWaveform,
    *,
    baseline: BaselineResult | None = None,
    start_time_s: float = 0.0005,
    min_height_mV: float = 8.0,
    min_distance_samples: int = 150,
    prominence_mV: float = 5.0,
    peak_pick_sigma_mV: float = 1.0,
) -> EchoPeaksResult:
    """Detect positive echo maxima after the initial ring-down."""
    baseline = baseline or estimate_baseline(waveform)
    amplitude_mV = (waveform.voltage_v - baseline.mean_v) * 1e3

    mask = waveform.time_s > start_time_s
    trace = amplitude_mV[mask]
    times_ms = waveform.time_s[mask] * 1e3

    peaks, props = find_peaks(
        trace,
        height=min_height_mV,
        distance=min_distance_samples,
        prominence=prominence_mV,
    )
    heights_mV = props["peak_heights"]
    sigma_mV = np.full_like(heights_mV, math.hypot(baseline.rms_v * 1e3, peak_pick_sigma_mV), dtype=float)

    return EchoPeaksResult(
        time_ms=times_ms[peaks],
        peak_mV=heights_mV + baseline.mean_v * 1e3,
        amplitude_mV=heights_mV,
        sigma_amplitude_mV=sigma_mV,
        baseline_mV=baseline.mean_v * 1e3,
        baseline_rms_mV=baseline.rms_v * 1e3,
    )


def fit_t2_envelope(
    time_ms: np.ndarray,
    amplitude_mV: np.ndarray,
    sigma_amplitude_mV: np.ndarray,
) -> T2FitResult:
    """Fit the Meiboom-Gill echo envelope to a single exponential with no offset."""
    time_ms = np.asarray(time_ms, dtype=float)
    amplitude_mV = np.asarray(amplitude_mV, dtype=float)
    sigma_amplitude_mV = np.asarray(sigma_amplitude_mV, dtype=float)

    popt, pcov = curve_fit(
        t2_envelope_model,
        time_ms,
        amplitude_mV,
        sigma=sigma_amplitude_mV,
        absolute_sigma=True,
        p0=(float(np.max(amplitude_mV)), 45.0),
        bounds=([0.0, 1.0], [500.0, 500.0]),
        maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))
    return T2FitResult(
        amplitude_mV=float(popt[0]),
        t2_ms=float(popt[1]),
        sigma_amplitude_mV=float(perr[0]),
        sigma_t2_ms=float(perr[1]),
        covariance=pcov,
    )
