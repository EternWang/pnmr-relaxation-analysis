"""Reproducible analysis utilities for a pulsed-NMR (PNMR) lab project."""

from .io import ScopeWaveform, read_scope_csv, load_t1_dataset, load_zero_crossing_metadata
from .t1 import fit_t1_inversion_recovery, propagate_t1_from_zero_crossing
from .t2 import estimate_baseline, extract_echo_peaks, fit_t2_envelope

__all__ = [
    "ScopeWaveform",
    "read_scope_csv",
    "load_t1_dataset",
    "load_zero_crossing_metadata",
    "fit_t1_inversion_recovery",
    "propagate_t1_from_zero_crossing",
    "estimate_baseline",
    "extract_echo_peaks",
    "fit_t2_envelope",
]
