# CL1 Biomarker Benchmark

**A reproducible biomarker and machine learning benchmark for the Cortical Labs CL1 ecosystem**

This project provides an experimental benchmark to compare **neural biomarkers extracted from the Cortical Labs CL API / CL SDK workflow** with **classical machine learning models** under a shared evaluation protocol.

It is designed as a research and engineering framework for exploring questions such as:
- which neural biomarkers best discriminate stimulus conditions,
- whether simulator and replay behavior align with observed recordings,
- how biomarker-based representations compare against classical ML baselines,
- and which biomarkers may anticipate adaptive performance improvements over time.

This repository is **not an official Cortical Labs project**.  
It is an independent research benchmark built to be **compatible with the Cortical Labs CL1 ecosystem**, especially the public developer tooling around **`cl-sdk`** and **`cl-api-doc`**.

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
```
## Relation to the Cortical Labs ecosystem

This benchmark is designed to work alongside the public Cortical Labs developer ecosystem, including:

- `cl-sdk` — the Cortical Labs CL API simulator for local development
- `cl-api-doc` — the public developer notebooks and API examples for the CL1 platform

Official repositories:
- Cortical Labs GitHub organization: https://github.com/Cortical-Labs
- CL SDK: https://github.com/Cortical-Labs/cl-sdk
- CL API docs: https://github.com/Cortical-Labs/cl-api-doc

The goal of this project is to extend that ecosystem with a reproducible evaluation layer focused on:
- biomarker extraction,
- model comparison,
- reproducibility analysis,
- and benchmark reporting.