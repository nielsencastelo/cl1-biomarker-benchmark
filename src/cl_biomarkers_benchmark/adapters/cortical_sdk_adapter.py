from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CLRecordingBundle:
    label: str
    file_path: Path
    metadata: dict[str, Any]


class CorticalSDKAdapter:
    """Adaptador para uso com gravações HDF5 / replay do ecossistema Cortical Labs.

    Estratégia:
    - `mode=replay`: usa `RecordingView(file_path)` para extrair biomarcadores de um HDF5.
    - `mode=sdk`: tenta abrir `cl`/`RecordingView` do ecossistema oficial, mas sem acoplar o restante do pipeline.

    O adapter foi desenhado para falhar de forma explícita quando `cl-sdk` não estiver instalado.
    """

    @staticmethod
    def require_cl_sdk() -> Any:
        try:
            import cl  # type: ignore
            from cl import RecordingView  # type: ignore
        except Exception as exc:  # pragma: no cover - ambiente pode não ter sdk
            raise ImportError(
                "cl-sdk / CL API não encontrado. Instale com `pip install cl-sdk` "
                "ou execute em um ambiente com o SDK oficial da Cortical Labs."
            ) from exc
        return cl, RecordingView

    def open_recording(self, file_path: str | Path):
        _, RecordingView = self.require_cl_sdk()
        return RecordingView(str(file_path))

    def load_replay_manifest(self, file_paths: list[str | Path], labels: list[str]) -> pd.DataFrame:
        if len(file_paths) != len(labels):
            raise ValueError("file_paths e labels precisam ter o mesmo tamanho")
        rows = []
        for i, (fp, label) in enumerate(zip(file_paths, labels)):
            rows.append(
                {
                    "sample_id": f"replay_{i}",
                    "label": label,
                    "file_path": str(fp),
                }
            )
        return pd.DataFrame(rows)

    def basic_recording_summary(self, file_path: str | Path) -> dict[str, Any]:
        recording = self.open_recording(file_path)
        summary = {
            "file_path": str(file_path),
            "n_spikes": len(recording.spikes) if getattr(recording, "spikes", None) is not None else None,
            "n_stims": len(recording.stims) if getattr(recording, "stims", None) is not None else None,
            "data_streams": list(recording.data_streams.keys()) if getattr(recording, "data_streams", None) else [],
        }
        try:
            recording.close()
        except Exception:
            pass
        return summary

    @staticmethod
    def spikes_table_to_dense(spikes_table: Any, n_channels: int | None = None, n_bins: int = 128) -> np.ndarray:
        """Converte uma tabela de spikes do RecordingView em matriz canal x tempo simplificada.

        Este método é intencionalmente conservador. A estrutura exata pode variar conforme gravação.
        """
        if len(spikes_table) == 0:
            return np.zeros((n_channels or 1, n_bins), dtype=float)

        channels = np.array([int(row["channel"]) for row in spikes_table])
        timestamps = np.array([int(row["timestamp"]) for row in spikes_table])
        if n_channels is None:
            n_channels = int(channels.max()) + 1
        start, end = timestamps.min(), timestamps.max() + 1
        bins = np.linspace(start, end, n_bins + 1)
        dense = np.zeros((n_channels, n_bins), dtype=float)
        for ch in np.unique(channels):
            mask = channels == ch
            hist, _ = np.histogram(timestamps[mask], bins=bins)
            dense[int(ch), :] = hist
        if dense.max() > 0:
            dense /= dense.max()
        return dense
