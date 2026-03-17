from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticRecording:
    label: str
    spikes: np.ndarray  # shape: (n_windows, n_channels, n_bins)
    timestamps: np.ndarray


class SyntheticAdapter:
    """Gera atividade sintética com padrões separáveis para desenvolvimento local.

    Não substitui dados reais do ecossistema Cortical Labs. Serve para validar o pipeline.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_recordings(
        self,
        labels: list[str],
        n_samples_per_class: int,
        n_channels: int,
        n_bins: int = 40,
        noise_std: float = 0.08,
    ) -> list[SyntheticRecording]:
        recordings: list[SyntheticRecording] = []

        label_profiles = {
            labels[0]: {"base_rate": 0.10, "sync": 0.12, "burst_bin": 8},
            labels[1]: {"base_rate": 0.22, "sync": 0.30, "burst_bin": 18},
            labels[2]: {"base_rate": 0.16, "sync": 0.52, "burst_bin": 28},
        }

        for label in labels:
            profile = label_profiles[label]
            spikes = []
            timestamps = []
            for i in range(n_samples_per_class):
                activity = self.rng.binomial(
                    1, profile["base_rate"], size=(n_channels, n_bins)
                ).astype(float)

                sync_mask = self.rng.random(size=(n_bins,)) < profile["sync"]
                activity[:, sync_mask] = 1.0

                burst_bin = max(0, min(n_bins - 3, profile["burst_bin"] + self.rng.integers(-2, 3)))
                activity[:, burst_bin : burst_bin + 3] = np.maximum(
                    activity[:, burst_bin : burst_bin + 3],
                    self.rng.binomial(1, 0.85, size=(n_channels, 3)),
                )

                noise = self.rng.normal(0, noise_std, size=activity.shape)
                activity = np.clip(activity + noise, 0.0, 1.0)
                spikes.append(activity)
                timestamps.append(i)

            recordings.append(
                SyntheticRecording(
                    label=label,
                    spikes=np.stack(spikes, axis=0),
                    timestamps=np.array(timestamps),
                )
            )
        return recordings

    @staticmethod
    def to_dataframe(recordings: list[SyntheticRecording]) -> pd.DataFrame:
        rows = []
        for rec in recordings:
            for i, sample in enumerate(rec.spikes):
                rows.append(
                    {
                        "sample_id": f"{rec.label}_{i}",
                        "label": rec.label,
                        "spikes": sample,
                        "timestamp": rec.timestamps[i],
                    }
                )
        return pd.DataFrame(rows)
