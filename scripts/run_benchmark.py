from __future__ import annotations

import argparse
from pathlib import Path

from cl_biomarkers_benchmark.benchmarks.experiment import BenchmarkExperiment
from cl_biomarkers_benchmark.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CL Biomarkers Benchmark")
    parser.add_argument("--config", required=True, help="Arquivo YAML de configuração")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["synthetic", "replay", "sdk"],
        help="Modo de execução",
    )
    parser.add_argument(
        "--recording",
        action="append",
        default=[],
        help="Caminho de gravação HDF5. Pode repetir várias vezes.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Rótulo correspondente a cada gravação. Pode repetir várias vezes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    experiment = BenchmarkExperiment(cfg)
    artifacts = experiment.run(
        mode=args.mode,
        recording_paths=args.recording or None,
        labels=args.label or None,
    )
    print("Benchmark concluído.")
    print(f"Dataset: {artifacts.dataset_path}")
    print(f"Metrics: {artifacts.metrics_path}")
    print(f"Summary: {artifacts.summary_path}")


if __name__ == "__main__":
    main()
