from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np


@dataclass
class BiomarkerResult:
    features: dict[str, float]
    meta: dict[str, Any]


def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else 0.0


def _safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size else 0.0


def _binary_entropy(p: float) -> float:
    p = float(np.clip(p, 1e-8, 1 - 1e-8))
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def _spike_intervals(signal: np.ndarray) -> np.ndarray:
    idx = np.where(signal > 0)[0]
    if len(idx) < 2:
        return np.array([], dtype=float)
    return np.diff(idx).astype(float)


def compute_biomarkers_from_dense_spikes(spikes: np.ndarray, label: str | None = None) -> BiomarkerResult:
    """Extrai biomarcadores simples de uma matriz canal x tempo.

    `spikes` é uma matriz densa canal x bin. Valores podem ser contagens ou sinais normalizados.
    """
    if spikes.ndim != 2:
        raise ValueError("spikes precisa ter shape (n_channels, n_bins)")

    spikes = np.asarray(spikes, dtype=float)
    binary = (spikes > 0).astype(float)

    per_channel_rate = binary.mean(axis=1)
    population_rate = binary.mean(axis=0)
    intervals = np.concatenate([_spike_intervals(ch) for ch in binary if ch.sum() > 1])

    synchrony_proxy = float(np.mean(np.sum(binary, axis=0) / max(1, binary.shape[0])))
    burst_threshold = np.quantile(population_rate, 0.85)
    burst_bins = population_rate >= burst_threshold
    burst_count = int(np.sum(np.diff(np.r_[0, burst_bins.astype(int), 0]) == 1))
    burst_density = float(burst_count / max(1, binary.shape[1]))

    corr = np.corrcoef(binary)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    graph = nx.from_numpy_array((corr > 0.35).astype(int))
    degrees = np.array([d for _, d in graph.degree()], dtype=float) if graph.number_of_nodes() else np.array([0.0])
    clustering = nx.average_clustering(graph) if graph.number_of_nodes() > 1 else 0.0

    features = {
        "firing_rate_mean": _safe_mean(per_channel_rate),
        "firing_rate_std": _safe_std(per_channel_rate),
        "population_rate_mean": _safe_mean(population_rate),
        "population_rate_std": _safe_std(population_rate),
        "isi_mean": _safe_mean(intervals),
        "isi_std": _safe_std(intervals),
        "isi_cv": float(_safe_std(intervals) / _safe_mean(intervals)) if intervals.size and _safe_mean(intervals) > 0 else 0.0,
        "synchrony_proxy": synchrony_proxy,
        "burst_count": float(burst_count),
        "burst_density": burst_density,
        "activity_entropy_mean": _safe_mean(np.array([_binary_entropy(p) for p in per_channel_rate])),
        "connectivity_mean": float(corr[corr > 0].mean()) if np.any(corr > 0) else 0.0,
        "connectivity_std": float(corr[corr > 0].std()) if np.any(corr > 0) else 0.0,
        "graph_degree_mean": _safe_mean(degrees),
        "graph_degree_std": _safe_std(degrees),
        "graph_clustering": float(clustering),
    }
    return BiomarkerResult(features=features, meta={"label": label} if label else {})


def compute_biomarkers_with_cl_recording(recording: Any, cfg: dict[str, Any]) -> BiomarkerResult:
    """Usa APIs oficiais quando disponíveis, complementando com proxies estáveis.

    Espera um objeto compatível com `RecordingView` do ecossistema Cortical Labs.
    """
    firing_cfg = cfg["biomarkers"]["firing_stats"]
    burst_cfg = cfg["biomarkers"]["network_bursts"]
    fc_cfg = cfg["biomarkers"]["functional_connectivity"]

    firing = recording.analyse_firing_stats(bin_size_sec=firing_cfg["bin_size_sec"])
    bursts = recording.analyse_network_bursts(
        bin_size_sec=burst_cfg["bin_size_sec"],
        onset_freq_hz=burst_cfg["onset_freq_hz"],
        offset_freq_hz=burst_cfg["offset_freq_hz"],
        min_active_channels=burst_cfg.get("min_active_channels"),
    )
    connectivity = recording.analyse_functional_connectivity(
        bin_size_sec=fc_cfg["bin_size_sec"],
        correlation_threshold=fc_cfg["correlation_threshold"],
    )

    # Como a estrutura interna dos resultados pode evoluir, usamos getattr com fallback.
    features = {
        "firing_rate_mean": float(getattr(firing, "population_rate_mean_hz", 0.0) or 0.0),
        "firing_rate_std": float(getattr(firing, "population_rate_std_hz", 0.0) or 0.0),
        "isi_mean": float(getattr(firing, "isi_mean_sec", 0.0) or 0.0),
        "isi_std": float(getattr(firing, "isi_std_sec", 0.0) or 0.0),
        "burst_count": float(getattr(bursts, "burst_count", 0.0) or 0.0),
        "burst_density": float(getattr(bursts, "burst_count", 0.0) or 0.0),
        "connectivity_mean": float(getattr(connectivity, "average_edge_weight", 0.0) or 0.0),
        "graph_clustering": float(getattr(connectivity, "clustering_coefficient", 0.0) or 0.0),
        "graph_degree_mean": float(getattr(connectivity, "total_edge_weight", 0.0) or 0.0),
        "modularity_index": float(getattr(connectivity, "modularity_index", 0.0) or 0.0),
        "betweenness_max": float(getattr(connectivity, "max_betweenness_centrality", 0.0) or 0.0),
    }
    return BiomarkerResult(features=features, meta={})
