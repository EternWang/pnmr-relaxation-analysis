from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import curve_fit

from .models import t1_magnitude_model


@dataclass(slots=True)
class T1ZeroCrossingResult:
    tau0_ms: float
    sigma_tau0_ms: float
    t1_ms: float
    sigma_t1_ms: float


@dataclass(slots=True)
class T1FitResult:
    offset_mV: float
    amplitude_mV: float
    t1_ms: float
    sigma_offset_mV: float
    sigma_amplitude_mV: float
    sigma_t1_ms: float
    covariance: np.ndarray


def propagate_t1_from_zero_crossing(tau0_ms: float, sigma_tau0_ms: float) -> T1ZeroCrossingResult:
    """Compute T1 and its propagated uncertainty from tau0 / ln 2."""
    t1_ms = tau0_ms / math.log(2.0)
    sigma_t1_ms = sigma_tau0_ms / math.log(2.0)
    return T1ZeroCrossingResult(
        tau0_ms=tau0_ms,
        sigma_tau0_ms=sigma_tau0_ms,
        t1_ms=t1_ms,
        sigma_t1_ms=sigma_t1_ms,
    )


def fit_t1_inversion_recovery(
    delays_ms: np.ndarray,
    amplitudes_mV: np.ndarray,
    sigma_mV: np.ndarray,
) -> T1FitResult:
    """Weighted nonlinear fit to the magnitude-detector inversion-recovery model."""
    delays_ms = np.asarray(delays_ms, dtype=float)
    amplitudes_mV = np.asarray(amplitudes_mV, dtype=float)
    sigma_mV = np.asarray(sigma_mV, dtype=float)

    p0 = (0.0, float(np.max(amplitudes_mV)), 50.0)
    bounds = ([-50.0, 0.0, 1.0], [50.0, 500.0, 500.0])

    popt, pcov = curve_fit(
        t1_magnitude_model,
        delays_ms,
        amplitudes_mV,
        sigma=sigma_mV,
        absolute_sigma=True,
        p0=p0,
        bounds=bounds,
        maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))
    return T1FitResult(
        offset_mV=float(popt[0]),
        amplitude_mV=float(popt[1]),
        t1_ms=float(popt[2]),
        sigma_offset_mV=float(perr[0]),
        sigma_amplitude_mV=float(perr[1]),
        sigma_t1_ms=float(perr[2]),
        covariance=pcov,
    )
