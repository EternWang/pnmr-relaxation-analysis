from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .io import load_t1_dataset, load_zero_crossing_metadata, read_scope_csv
from .plotting import plot_results_dashboard, plot_t1_fit, plot_t2_fit, plot_waveform_with_peaks
from .t1 import fit_t1_inversion_recovery, propagate_t1_from_zero_crossing
from .t2 import estimate_baseline, extract_echo_peaks, fit_t2_envelope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full PNMR analysis workflow.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Directory containing raw input data.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Directory for processed tables.")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"), help="Directory for figures.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # T1 analysis
    t1_df = load_t1_dataset(args.data_dir / "t1_inversion_recovery.csv")
    zero_meta = load_zero_crossing_metadata(args.data_dir / "t1_zero_crossing.json")
    t1_zero = propagate_t1_from_zero_crossing(**zero_meta)
    t1_fit = fit_t1_inversion_recovery(
        t1_df["delay_ms"].to_numpy(),
        t1_df["amplitude_mV"].to_numpy(),
        t1_df["sigma_amplitude_mV"].to_numpy(),
    )
    plot_t1_fit(
        t1_df["delay_ms"].to_numpy(),
        t1_df["amplitude_mV"].to_numpy(),
        t1_df["sigma_amplitude_mV"].to_numpy(),
        (t1_fit.offset_mV, t1_fit.amplitude_mV, t1_fit.t1_ms),
        args.figures_dir / "t1_fit.png",
    )

    # T2 analysis
    waveform = read_scope_csv(args.data_dir / "DS0001A.CSV")
    baseline = estimate_baseline(waveform)
    peaks = extract_echo_peaks(waveform, baseline=baseline)
    t2_fit = fit_t2_envelope(peaks.time_ms, peaks.amplitude_mV, peaks.sigma_amplitude_mV)

    pd.DataFrame(
        {
            "echo_number": range(1, len(peaks.time_ms) + 1),
            "time_ms": peaks.time_ms,
            "peak_mV": peaks.peak_mV,
            "amplitude_mV": peaks.amplitude_mV,
            "sigma_amplitude_mV": peaks.sigma_amplitude_mV,
        }
    ).to_csv(args.processed_dir / "t2_echo_peaks.csv", index=False)

    plot_t2_fit(
        peaks.time_ms,
        peaks.amplitude_mV,
        peaks.sigma_amplitude_mV,
        (t2_fit.amplitude_mV, t2_fit.t2_ms),
        args.figures_dir / "t2_fit.png",
    )
    plot_waveform_with_peaks(
        waveform.time_ms,
        (waveform.voltage_v - baseline.mean_v) * 1e3,
        peaks.time_ms,
        peaks.amplitude_mV,
        args.figures_dir / "t2_waveform_with_peaks.png",
    )
    plot_results_dashboard(
        t1_df["delay_ms"].to_numpy(),
        t1_df["amplitude_mV"].to_numpy(),
        t1_df["sigma_amplitude_mV"].to_numpy(),
        (t1_fit.offset_mV, t1_fit.amplitude_mV, t1_fit.t1_ms),
        peaks.time_ms,
        peaks.amplitude_mV,
        peaks.sigma_amplitude_mV,
        (t2_fit.amplitude_mV, t2_fit.t2_ms),
        args.figures_dir / "results_dashboard.png",
    )

    summary = {
        "t1_zero_crossing_ms": round(t1_zero.t1_ms, 3),
        "t1_zero_crossing_sigma_ms": round(t1_zero.sigma_t1_ms, 3),
        "t1_fit_ms": round(t1_fit.t1_ms, 3),
        "t1_fit_sigma_ms": round(t1_fit.sigma_t1_ms, 3),
        "t2_fit_ms": round(t2_fit.t2_ms, 3),
        "t2_fit_sigma_ms": round(t2_fit.sigma_t2_ms, 3),
        "baseline_mV": round(peaks.baseline_mV, 3),
        "baseline_rms_mV": round(peaks.baseline_rms_mV, 3),
        "n_echo_peaks": int(len(peaks.time_ms)),
    }
    (args.processed_dir / "summary_results.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
