# PNMR Analysis Showcase

A GitHub-ready Python project built around a **pulsed nuclear magnetic resonance (PNMR)** experiment from the UCSB upper-division physics lab.  
The goal of this repository is to showcase **data analysis, scientific Python, reproducibility, uncertainty handling, and clean project structure**—not just to store a final report.

## What this repository demonstrates

- Parsing a real oscilloscope waveform export (`DS0001A.CSV`)
- Estimating baseline noise from pre-trigger samples
- Detecting echo maxima with signal-processing tools
- Performing weighted nonlinear least-squares fits for `T₁` and `T₂`
- Propagating uncertainty from a manually measured zero crossing
- Generating publication-style figures automatically
- Organizing an experiment as a reusable Python package with tests and a notebook

## Key quantitative results

| Quantity | Result |
|---|---:|
| Zero-beat resonance | `f₀ ≈ 15.146 MHz` |
| Static magnetic field | `B₀ ≈ 3.557 kG` (`0.3557 T`) |
| `T₁` from zero crossing | `51.22 ± 3.61 ms` |
| `T₁` from weighted nonlinear fit | `52.10 ± 1.08 ms` |
| `T₂` from MG waveform fit | `44.55 ± 1.05 ms` |
| Baseline RMS noise | `0.891 mV` |
| Echo peaks detected | `36` |

## Example outputs

### Analysis dashboard
![Results dashboard](figures/results_dashboard.png)

### Echo detection on the raw waveform
![Waveform with detected peaks](figures/t2_waveform_with_peaks.png)

## Repository structure

```text
pnmr-github-showcase/
├── .github/workflows/ci.yml        # GitHub Actions test workflow
├── data/
│   ├── raw/
│   │   ├── DS0001A.CSV             # Meiboom–Gill waveform export
│   │   ├── part_a_cursor_measurements.csv
│   │   ├── t1_inversion_recovery.csv
│   │   └── t1_zero_crossing.json
│   └── processed/
│       ├── summary_results.json
│       └── t2_echo_peaks.csv
├── figures/
│   ├── results_dashboard.png
│   ├── t1_fit.png
│   ├── t2_fit.png
│   └── t2_waveform_with_peaks.png
├── notebooks/
│   └── 01_pnmr_analysis_showcase.ipynb
├── report/
│   ├── PNMR_report.pdf
│   └── PNMR_report_source.tex
├── scripts/
│   └── run_full_analysis.py
├── src/pnmr_showcase/
│   ├── __init__.py
│   ├── cli.py
│   ├── io.py
│   ├── models.py
│   ├── plotting.py
│   ├── t1.py
│   └── t2.py
├── tests/
│   └── test_analysis.py
├── LICENSE
├── Makefile
├── pyproject.toml
└── README.md
```

## Reproducibility

### Option 1: local virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python scripts/run_full_analysis.py
pytest -q
```

### Option 2: make targets
```bash
make install
make analysis
make test
```

## Analysis notes

### `T₁` workflow
The inversion-recovery dataset is stored in `data/raw/t1_inversion_recovery.csv`.  
This repository reports two complementary `T₁` estimates:

1. **Primary value** from a zero-crossing measurement:  
   `T₁ = τ₀ / ln 2`
2. **Cross-check** from a weighted nonlinear fit to the magnitude-detector model:  
   `A(τ) = A_off + A₀ |1 - 2 exp(-τ / T₁)|`

This is a nice example of combining **manual experimental judgment** with **modern regression tools**.

### `T₂` workflow
The `T₂` analysis uses the raw oscilloscope export directly:

1. Estimate the pre-trigger baseline mean and RMS noise
2. Subtract the baseline
3. Detect positive echo maxima with `scipy.signal.find_peaks`
4. Assign a per-peak uncertainty from the measured noise and a peak-picking term
5. Fit the Meiboom–Gill echo envelope to  
   `A(t) = A₀ exp(-t / T₂)`

That gives a fully scriptable, reproducible `T₂` pipeline.

## Why this is portfolio-friendly

This repo shows more than “I can write a lab report.” It shows that I can:

- turn messy experimental data into a reproducible pipeline
- write modular scientific Python
- document assumptions and uncertainty sources honestly
- package an experiment in a way that can live on GitHub

## Related files

- Final report PDF: `report/PNMR_report.pdf`
- Executed notebook: `notebooks/01_pnmr_analysis_showcase.ipynb`

---
Built from a UCSB Physics PNMR experiment by **Hongyu Wang** and **Miles Bondoc**.
