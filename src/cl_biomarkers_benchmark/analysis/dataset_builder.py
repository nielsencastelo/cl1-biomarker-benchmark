from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cl_biomarkers_benchmark.adapters.cortical_sdk_adapter import CorticalSDKAdapter
from cl_biomarkers_benchmark.analysis.biomarkers import (
    compute_biomarkers_from_dense_spikes,
    compute_biomarkers_with_cl_recording,
)


class DatasetBuilder:
    def build_from_synthetic_df(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            result = compute_biomarkers_from_dense_spikes(row["spikes"], label=row["label"])
            rows.append({"sample_id": row["sample_id"], "label": row["label"], **result.features})
        return pd.DataFrame(rows)

    def build_from_replay_manifest(self, manifest_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
        adapter = CorticalSDKAdapter()
        rows: list[dict[str, Any]] = []
        for _, row in manifest_df.iterrows():
            file_path = Path(row["file_path"])
            recording = adapter.open_recording(file_path)
            result = compute_biomarkers_with_cl_recording(recording, cfg)
            try:
                recording.close()
            except Exception:
                pass
            rows.append({"sample_id": row["sample_id"], "label": row["label"], **result.features})
        return pd.DataFrame(rows)
