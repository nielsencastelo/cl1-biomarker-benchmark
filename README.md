# CL Biomarkers Benchmark

An experimental benchmark to compare **biological substrate via the Cortical Labs CL API / CL SDK** with **classical ML models** using biomarkers derived from neural activity.

## Objective

To answer questions such as:

- In which simple adaptive tasks does the biological system show a relative advantage?
- Does the simulator (`cl-sdk`) accurately represent the behavior observed in recording replays?
- Which neural biomarkers anticipate performance improvement?
- How do classical ML models behave under the same experimental protocol?

## What this project delivers

- A ready-to-use structure for **simulation**, **replay**, and **real execution** within the Cortical Labs ecosystem.
- Biomarker extraction from HDF5 recordings using `RecordingView`.
- ML baselines (Logistic Regression, Random Forest, Gradient Boosting, MLP).
- A comparative benchmark with metrics such as:
  - accuracy,
  - macro F1,
  - log loss,
  - Brier score,
  - training time,
  - inference time per sample,
  - fold stability.
- Automatic generation of a Markdown report for sharing results.

## Suggested scientific scope

### Main task
Temporal stimulus pattern classification (A/B/C), comparing:

1. neural response / biomarkers extracted from `cl-sdk` or HDF5 replay,
2. classical ML models trained on the same biomarkers,
3. a synthetic baseline for local development without hardware.

### Hypothesis
Neural biomarkers such as **firing rate**, **burst density**, **ISI CV**, **synchrony proxy**, and **functional connectivity** carry enough signal to discriminate states/stimuli; furthermore, the evolution of these biomarkers over time may anticipate system adaptation.

---

## Architecture

```text
configs/
  experiment.temporal_patterns.yaml
scripts/
  run_benchmark.py
src/cl_biomarkers_benchmark/
  adapters/
    cortical_sdk_adapter.py
    synthetic_adapter.py
  analysis/
    biomarkers.py
    dataset_builder.py
  benchmarks/
    experiment.py
  ml/
    baselines.py
    metrics.py
  reporting/
    report.py
  utils/
    config.py