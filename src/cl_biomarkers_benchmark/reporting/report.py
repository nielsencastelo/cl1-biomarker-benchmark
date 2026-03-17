from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def save_metrics_json(results: list[dict], output_dir: Path) -> Path:
    path = output_dir / "metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return path


def save_summary_markdown(summary_df: pd.DataFrame, output_dir: Path, experiment_name: str, mode: str) -> Path:
    path = output_dir / "summary.md"
    best = summary_df.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {experiment_name}\n\n")
        f.write(f"Modo: `{mode}`\n\n")
        f.write("## Melhor modelo\n\n")
        f.write(
            f"- Modelo: **{best['model']}**\n"
            f"- F1 macro: **{best['f1_macro']:.4f}**\n"
            f"- Accuracy: **{best['accuracy']:.4f}**\n"
            f"- Log loss: **{best['log_loss']:.4f}**\n"
            f"- Brier score: **{best['brier_score']:.4f}**\n"
        )
        f.write("\n## Tabela comparativa\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Interpretação sugerida\n\n")
        f.write(
            "Modelos com maior F1 macro e menor log loss indicam melhor separação dos biomarcadores entre estímulos. "
            "Quando o modo for `replay` ou `sdk`, diferenças para o benchmark sintético podem indicar limitações do simulador "
            "ou maior variabilidade biológica.\n"
        )
    return path
