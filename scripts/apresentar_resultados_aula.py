from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import matplotlib.pyplot as plt

try:
    from scripts._bootstrap import bootstrap_src_path
except ModuleNotFoundError:  # pragma: no cover
    from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from ecg_classification.config import ProjectPaths


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def present_classification_results(paths: ProjectPaths) -> None:
    metrics_path = paths.reports / "tables" / "classical_ml_metrics.json"
    confusion_path = paths.reports / "tables" / "classical_ml_confusion_matrix.csv"

    if not metrics_path.exists() or not confusion_path.exists():
        print("Resultados de classificacao nao encontrados.")
        print("Execute antes: python scripts/train_classical_ml.py")
        return

    metrics = load_json(metrics_path)
    confusion_rows = load_csv_rows(confusion_path)
    class_labels = [key for key in confusion_rows[0].keys() if key]

    print_header("RESULTADOS DO CLASSIFICADOR DE BATIMENTOS")
    print("Base: MIT-BIH Arrhythmia Database")
    print("Modelo: Random Forest balanceado")
    print(f"Acuracia geral:      {metrics['accuracy']:.2%}")
    print(f"Precisao macro:      {metrics['precision_macro']:.2%}")
    print(f"Recall macro:        {metrics['recall_macro']:.2%}")
    print(f"F1-score macro:      {metrics['f1_macro']:.2%}")

    total_samples = 0
    correct_samples = 0
    per_class_totals: dict[str, int] = {}
    confusion_matrix: list[list[int]] = []

    for row in confusion_rows:
        real_label = row[""]
        values = [int(row[label]) for label in class_labels]
        confusion_matrix.append(values)
        row_total = sum(values)
        total_samples += row_total
        per_class_totals[real_label] = row_total
        if real_label in class_labels:
            correct_samples += int(row[real_label])

    print(f"Total de batimentos avaliados: {total_samples}")
    print(f"Predicoes corretas:            {correct_samples}")

    print("\nMatriz de confusao:")
    header = "      " + "".join(f"{label:>6}" for label in class_labels)
    print(header)
    for row in confusion_rows:
        real_label = row[""]
        values = "".join(f"{int(row[label]):>6}" for label in class_labels)
        print(f"{real_label:>2}{values}")

    print("\nDistribuicao de amostras por classe real:")
    ordered_classes = sorted(per_class_totals.items(), key=lambda item: item[1], reverse=True)
    for label, count in ordered_classes:
        print(f"Classe {label}: {count} batimentos")

    output_dir = paths.reports / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metric_names = list(metrics.keys())
    metric_values = [metrics[name] * 100 for name in metric_names]
    axes[0].bar(metric_names, metric_values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    axes[0].set_title("Metricas de classificacao (%)")
    axes[0].set_ylabel("Percentual")
    axes[0].set_ylim(0, 100)
    axes[0].tick_params(axis="x", rotation=20)

    ordered_labels = [label for label, _ in ordered_classes]
    ordered_counts = [count for _, count in ordered_classes]
    axes[1].bar(ordered_labels, ordered_counts, color="#4c78a8")
    axes[1].set_title("Distribuicao das classes reais")
    axes[1].set_ylabel("Numero de batimentos")
    axes[1].tick_params(axis="x", rotation=0)

    fig.tight_layout()
    figure_path = output_dir / "apresentacao_classificacao.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigura salva em: {figure_path}")


def present_rpeak_results(paths: ProjectPaths) -> None:
    metrics_path = paths.reports / "tables" / "rpeak_detection_metrics.csv"
    if not metrics_path.exists():
        print("\nResultados de deteccao de picos R nao encontrados.")
        print("Execute antes: python scripts/evaluate_rpeak_detection.py")
        return

    table = load_csv_rows(metrics_path)
    numeric_fields = {
        "precision",
        "recall",
        "f1",
        "mean_absolute_error_ms",
    }
    for row in table:
        for field in numeric_fields:
            row[field] = float(row[field])

    print_header("RESULTADOS DA DETECCAO DE PICOS R")
    print(f"Quantidade de registros avaliados: {len(table)}")
    precision_mean = sum(row["precision"] for row in table) / len(table)
    recall_mean = sum(row["recall"] for row in table) / len(table)
    f1_mean = sum(row["f1"] for row in table) / len(table)
    mae_ms_mean = sum(row["mean_absolute_error_ms"] for row in table) / len(table)
    print(f"Precisao media: {precision_mean:.2%}")
    print(f"Recall medio:   {recall_mean:.2%}")
    print(f"F1 medio:       {f1_mean:.2%}")
    print(f"Erro medio medio (ms): {mae_ms_mean:.2f}")

    best_records = sorted(table, key=lambda row: row["f1"], reverse=True)[:5]
    worst_records = sorted(table, key=lambda row: row["f1"])[:5]

    print("\nMelhores registros na deteccao de picos R:")
    for row in best_records:
        print(
            f"Registro {row['record_id']}: "
            f"F1={row['f1']:.2%}, precisao={row['precision']:.2%}, recall={row['recall']:.2%}"
        )

    print("\nPiores registros na deteccao de picos R:")
    for row in worst_records:
        print(
            f"Registro {row['record_id']}: "
            f"F1={row['f1']:.2%}, precisao={row['precision']:.2%}, recall={row['recall']:.2%}"
        )

    output_dir = paths.reports / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ordered = sorted(table, key=lambda row: row["f1"], reverse=True)
    ax.bar([row["record_id"] for row in ordered], [row["f1"] * 100 for row in ordered], color="#59a14f")
    ax.set_title("F1 por registro na deteccao de picos R")
    ax.set_xlabel("Registro")
    ax.set_ylabel("F1-score (%)")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()

    figure_path = output_dir / "apresentacao_rpeaks.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigura salva em: {figure_path}")


def main() -> None:
    paths = ProjectPaths()
    print_header("RESUMO DIDATICO DOS RESULTADOS DO PROJETO")
    print("Este script foi criado para apresentar os resultados de forma simples em aula.")

    present_classification_results(paths)
    present_rpeak_results(paths)

    print_header("ENCERRAMENTO")
    print("Arquivos gerados podem ser usados diretamente em slides ou demonstracoes.")


if __name__ == "__main__":
    main()
