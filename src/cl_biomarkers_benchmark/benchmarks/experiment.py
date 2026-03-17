from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cl_biomarkers_benchmark.adapters.cortical_sdk_adapter import CorticalSDKAdapter
from cl_biomarkers_benchmark.adapters.synthetic_adapter import SyntheticAdapter
from cl_biomarkers_benchmark.analysis.dataset_builder import DatasetBuilder
from cl_biomarkers_benchmark.ml.baselines import run_cv
from cl_biomarkers_benchmark.reporting.report import save_metrics_json, save_summary_markdown
from cl_biomarkers_benchmark.utils.config import ensure_dir


@dataclass
class ExperimentArtifacts:
    dataset_path: Path
    metrics_path: Path
    summary_path: Path


class BenchmarkExperiment:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.output_dir = ensure_dir(cfg.get("output_dir", "artifacts"))
        self.dataset_builder = DatasetBuilder()

    def _dataset_synthetic(self) -> pd.DataFrame:
        adapter = SyntheticAdapter(seed=int(self.cfg.get("seed", 42)))
        recs = adapter.generate_recordings(
            labels=self.cfg["experiment"]["labels"],
            n_samples_per_class=int(self.cfg["synthetic"]["n_samples_per_class"]),
            n_channels=int(self.cfg["synthetic"]["n_channels"]),
            noise_std=float(self.cfg["synthetic"]["noise_std"]),
        )
        raw_df = adapter.to_dataframe(recs)
        return self.dataset_builder.build_from_synthetic_df(raw_df)

    def _dataset_replay(self, recording_paths: list[str], labels: list[str]) -> pd.DataFrame:
        adapter = CorticalSDKAdapter()
        manifest = adapter.load_replay_manifest(recording_paths, labels)
        return self.dataset_builder.build_from_replay_manifest(manifest, self.cfg)

    def run(self, mode: str, recording_paths: list[str] | None = None, labels: list[str] | None = None) -> ExperimentArtifacts:
        mode = mode.lower().strip()
        if mode == "synthetic":
            df = self._dataset_synthetic()
        elif mode in {"replay", "sdk"}:
            if not recording_paths or not labels:
                raise ValueError("Para mode=replay/sdk informe recording_paths e labels")
            df = self._dataset_replay(recording_paths, labels)
        else:
            raise ValueError(f"Modo inválido: {mode}")

        dataset_path = self.output_dir / "dataset.csv"
        df.to_csv(dataset_path, index=False)

        model_results = run_cv(df=df, cfg=self.cfg)
        summary_rows = []
        serializable_results = []
        for result in model_results:
            row = {
                "model": result.name,
                **result.metrics,
                "fit_time_sec": result.fit_time_sec,
                "predict_time_sec_per_sample": result.predict_time_sec,
            }
            summary_rows.append(row)
            serializable_results.append(row)

        summary_df = pd.DataFrame(summary_rows).sort_values(["f1_macro", "accuracy"], ascending=False)
        metrics_path = save_metrics_json(serializable_results, self.output_dir)
        summary_path = save_summary_markdown(
            summary_df,
            output_dir=self.output_dir,
            experiment_name=self.cfg["experiment"]["name"],
            mode=mode,
        )
        return ExperimentArtifacts(dataset_path=dataset_path, metrics_path=metrics_path, summary_path=summary_path)
