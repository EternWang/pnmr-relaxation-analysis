from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .models import t1_magnitude_model, t2_envelope_model


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )


def plot_t1_fit(
    delays_ms: np.ndarray,
    amplitudes_mV: np.ndarray,
    sigma_mV: np.ndarray,
    fit_params: tuple[float, float, float],
    save_path: str | Path,
) -> None:
    set_style()
    save_path = Path(save_path)
    grid = np.linspace(np.min(delays_ms), np.max(delays_ms), 500)
    fit = t1_magnitude_model(grid, *fit_params)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.errorbar(delays_ms, amplitudes_mV, yerr=sigma_mV, fmt="o", color="black", capsize=3, label="Measured amplitudes")
    ax.plot(grid, fit, color="0.35", lw=2.0, label="Weighted nonlinear fit")
    ax.set_xlabel("Delay $\\tau$ (ms)")
    ax.set_ylabel("Baseline-subtracted amplitude (mV)")
    ax.set_title("Inversion-recovery fit for $T_1$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_t2_fit(
    time_ms: np.ndarray,
    amplitude_mV: np.ndarray,
    sigma_mV: np.ndarray,
    fit_params: tuple[float, float],
    save_path: str | Path,
) -> None:
    set_style()
    save_path = Path(save_path)
    grid = np.linspace(np.min(time_ms), np.max(time_ms), 600)
    fit = t2_envelope_model(grid, *fit_params)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.errorbar(time_ms, amplitude_mV, yerr=sigma_mV, fmt="o", ms=3.8, color="black", capsize=2, label="Detected echo peaks")
    ax.plot(grid, fit, color="0.35", lw=2.0, label="Exponential fit")
    ax.set_xlabel("Echo time (ms)")
    ax.set_ylabel("Amplitude above baseline (mV)")
    ax.set_title("Meiboom-Gill echo-envelope fit for $T_2$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_waveform_with_peaks(
    time_ms: np.ndarray,
    amplitude_mV: np.ndarray,
    peak_times_ms: np.ndarray,
    peak_amplitudes_mV: np.ndarray,
    save_path: str | Path,
) -> None:
    set_style()
    save_path = Path(save_path)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(time_ms, amplitude_mV, color="0.20", lw=0.8, label="Baseline-subtracted waveform")
    ax.scatter(peak_times_ms, peak_amplitudes_mV, color="black", s=18, zorder=3, label="Detected peaks")
    ax.set_xlim(-2, 75)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude above baseline (mV)")
    ax.set_title("Echo detection on the raw scope waveform")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_results_dashboard(
    delays_ms: np.ndarray,
    t1_amplitudes_mV: np.ndarray,
    t1_sigma_mV: np.ndarray,
    t1_params: tuple[float, float, float],
    t2_time_ms: np.ndarray,
    t2_amplitudes_mV: np.ndarray,
    t2_sigma_mV: np.ndarray,
    t2_params: tuple[float, float],
    save_path: str | Path,
) -> None:
    set_style()
    save_path = Path(save_path)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))

    grid_t1 = np.linspace(np.min(delays_ms), np.max(delays_ms), 500)
    axes[0].errorbar(delays_ms, t1_amplitudes_mV, yerr=t1_sigma_mV, fmt="o", color="black", capsize=2)
    axes[0].plot(grid_t1, t1_magnitude_model(grid_t1, *t1_params), color="0.35", lw=2)
    axes[0].set_xlabel("Delay $\\tau$ (ms)")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].set_title("$T_1$ inversion recovery")

    grid_t2 = np.linspace(np.min(t2_time_ms), np.max(t2_time_ms), 600)
    axes[1].errorbar(t2_time_ms, t2_amplitudes_mV, yerr=t2_sigma_mV, fmt="o", ms=3.2, color="black", capsize=2)
    axes[1].plot(grid_t2, t2_envelope_model(grid_t2, *t2_params), color="0.35", lw=2)
    axes[1].set_xlabel("Echo time (ms)")
    axes[1].set_ylabel("Amplitude (mV)")
    axes[1].set_title("$T_2$ echo envelope")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
