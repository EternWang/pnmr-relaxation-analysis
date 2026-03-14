from pathlib import Path

from pnmr_showcase.io import load_t1_dataset, read_scope_csv
from pnmr_showcase.t1 import fit_t1_inversion_recovery, propagate_t1_from_zero_crossing
from pnmr_showcase.t2 import estimate_baseline, extract_echo_peaks, fit_t2_envelope


DATA = Path("data/raw")


def test_t1_zero_crossing_value():
    result = propagate_t1_from_zero_crossing(35.5, 2.5)
    assert abs(result.t1_ms - 51.215) < 0.05
    assert abs(result.sigma_t1_ms - 3.607) < 0.05


def test_t1_fit_value():
    df = load_t1_dataset(DATA / "t1_inversion_recovery.csv")
    fit = fit_t1_inversion_recovery(
        df["delay_ms"].to_numpy(),
        df["amplitude_mV"].to_numpy(),
        df["sigma_amplitude_mV"].to_numpy(),
    )
    assert 50.5 < fit.t1_ms < 53.5
    assert 0.5 < fit.sigma_t1_ms < 2.0


def test_t2_peak_detection_and_fit():
    waveform = read_scope_csv(DATA / "DS0001A.CSV")
    baseline = estimate_baseline(waveform)
    peaks = extract_echo_peaks(waveform, baseline=baseline)
    fit = fit_t2_envelope(peaks.time_ms, peaks.amplitude_mV, peaks.sigma_amplitude_mV)

    assert len(peaks.time_ms) == 36
    assert 117.5 < peaks.baseline_mV < 118.5
    assert 0.7 < peaks.baseline_rms_mV < 1.1
    assert 43.0 < fit.t2_ms < 46.5
    assert 0.5 < fit.sigma_t2_ms < 2.0
