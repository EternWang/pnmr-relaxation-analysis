from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .models import t1_magnitude_model, t2_envelope_model

BLUE = "#2F6B9A"
ORANGE = "#D97935"
GREEN = "#5B8C5A"
GRAY = "#4A5568"
LIGHT = "#EEF2F6"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9DEE7",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
        }
    )


def save_figure(fig: plt.Figure, save_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.errorbar(delays_ms, amplitudes_mV, yerr=sigma_mV, fmt="o", color=BLUE, ecolor="#7FA7C7", capsize=3, label="Measured amplitudes")
    ax.plot(grid, fit, color=ORANGE, lw=2.4, label="Weighted nonlinear fit")
    ax.set_xlabel("Delay $\\tau$ (ms)")
    ax.set_ylabel("Baseline-subtracted amplitude (mV)")
    ax.set_title("Inversion-recovery fit for T1")
    ax.text(
        0.03,
        0.95,
        f"T1 = {fit_params[2]:.2f} ms",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": LIGHT, "edgecolor": "#CBD5E1"},
    )
    ax.legend()
    save_figure(fig, save_path)


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

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.errorbar(time_ms, amplitude_mV, yerr=sigma_mV, fmt="o", ms=4, color=BLUE, ecolor="#7FA7C7", capsize=2, label="Detected echo peaks")
    ax.plot(grid, fit, color=ORANGE, lw=2.4, label="Exponential fit")
    ax.set_xlabel("Echo time (ms)")
    ax.set_ylabel("Amplitude above baseline (mV)")
    ax.set_title("Meiboom-Gill echo-envelope fit for T2")
    ax.text(
        0.03,
        0.95,
        f"T2 = {fit_params[1]:.2f} ms",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": LIGHT, "edgecolor": "#CBD5E1"},
    )
    ax.legend()
    save_figure(fig, save_path)


def plot_waveform_with_peaks(
    time_ms: np.ndarray,
    amplitude_mV: np.ndarray,
    peak_times_ms: np.ndarray,
    peak_amplitudes_mV: np.ndarray,
    save_path: str | Path,
) -> None:
    set_style()
    save_path = Path(save_path)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(time_ms, amplitude_mV, color=GRAY, lw=0.8, alpha=0.9, label="Baseline-subtracted waveform")
    ax.scatter(peak_times_ms, peak_amplitudes_mV, color=ORANGE, s=22, zorder=3, label=f"Detected peaks (n={len(peak_times_ms)})")
    ax.set_xlim(-2, 75)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude above baseline (mV)")
    ax.set_title("Echo detection on raw scope waveform")
    ax.legend(loc="upper right")
    save_figure(fig, save_path)


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
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    grid_t1 = np.linspace(np.min(delays_ms), np.max(delays_ms), 500)
    axes[0].errorbar(delays_ms, t1_amplitudes_mV, yerr=t1_sigma_mV, fmt="o", color=BLUE, ecolor="#7FA7C7", capsize=2)
    axes[0].plot(grid_t1, t1_magnitude_model(grid_t1, *t1_params), color=ORANGE, lw=2.4)
    axes[0].set_xlabel("Delay $\\tau$ (ms)")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].set_title("T1 inversion recovery")
    axes[0].text(
        0.04,
        0.94,
        f"T1 = {t1_params[2]:.2f} ms",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": LIGHT, "edgecolor": "#CBD5E1"},
    )

    grid_t2 = np.linspace(np.min(t2_time_ms), np.max(t2_time_ms), 600)
    axes[1].errorbar(t2_time_ms, t2_amplitudes_mV, yerr=t2_sigma_mV, fmt="o", ms=3.4, color=BLUE, ecolor="#7FA7C7", capsize=2)
    axes[1].plot(grid_t2, t2_envelope_model(grid_t2, *t2_params), color=ORANGE, lw=2.4)
    axes[1].set_xlabel("Echo time (ms)")
    axes[1].set_ylabel("Amplitude (mV)")
    axes[1].set_title("T2 echo envelope")
    axes[1].text(
        0.04,
        0.94,
        f"T2 = {t2_params[1]:.2f} ms",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": LIGHT, "edgecolor": "#CBD5E1"},
    )

    save_figure(fig, save_path)
