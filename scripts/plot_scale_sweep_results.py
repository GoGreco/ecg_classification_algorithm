"""Gera gráficos da varredura de escalas ``t`` da avaliação binária."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    output_dir = Path("reports/symbolic_lstm_scale_sweep_binary_test")
    results = pd.read_csv(output_dir / "repeated_fold_metrics.csv")
    summary = (
        results.groupby(["scale", "model"])
        .agg(
            fp_mean=("fp", "mean"),
            fp_std=("fp", "std"),
            fn_mean=("fn", "mean"),
            fn_std=("fn", "std"),
            sensitivity_mean=("sensitivity_non_N", "mean"),
            sensitivity_std=("sensitivity_non_N", "std"),
            specificity_mean=("specificity_N", "mean"),
            specificity_std=("specificity_N", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "scale_sweep_summary_for_plot.csv", index=False)

    colors = {"continuous": "#1f77b4", "symbolic": "#d62728"}
    labels = {"continuous": "Contínuo", "symbolic": "Simbólico"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    plots = [
        ("fp_mean", "fp_std", "Falsos positivos médios", "FP"),
        ("fn_mean", "fn_std", "Falsos negativos médios", "FN"),
        ("sensitivity_mean", "sensitivity_std", "Sensibilidade de não-N", "Sensibilidade"),
        ("specificity_mean", "specificity_std", "Especificidade de N", "Especificidade"),
    ]
    for axis, (mean_column, std_column, title, ylabel) in zip(axes.ravel(), plots):
        for model in ("continuous", "symbolic"):
            subset = summary[summary["model"] == model]
            axis.errorbar(
                subset["scale"],
                subset[mean_column],
                yerr=subset[std_column],
                marker="o",
                capsize=3,
                color=colors[model],
                label=labels[model],
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(sorted(summary["scale"].unique()))
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("t — amostras por bloco")
    axes[1, 1].set_xlabel("t — amostras por bloco")
    axes[0, 0].legend()
    fig.suptitle("Varredura de escalas: falsos positivos, falsos negativos e taxas")
    fig.tight_layout()
    fig.savefig(output_dir / "scale_sweep_fp_fn_rates.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    for model in ("continuous", "symbolic"):
        subset = summary[summary["model"] == model]
        axis.errorbar(
            subset["fp_mean"],
            subset["fn_mean"],
            xerr=subset["fp_std"],
            yerr=subset["fn_std"],
            marker="o",
            capsize=3,
            color=colors[model],
            label=labels[model],
        )
        for _, row in subset.iterrows():
            axis.annotate(f"t={int(row['scale'])}", (row["fp_mean"], row["fn_mean"]), xytext=(4, 4), textcoords="offset points")
    axis.set_xlabel("Falsos positivos médios")
    axis.set_ylabel("Falsos negativos médios")
    axis.set_title("Compromisso entre falsos positivos e falsos negativos")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "scale_sweep_fp_vs_fn.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for model in ("continuous", "symbolic"):
        subset = summary[summary["model"] == model]
        axis.errorbar(
            subset["scale"],
            subset["balanced_accuracy_mean"],
            yerr=subset["balanced_accuracy_std"],
            marker="o",
            capsize=3,
            color=colors[model],
            label=labels[model],
        )
    axis.set_xlabel("t — amostras por bloco")
    axis.set_ylabel("Balanced accuracy")
    axis.set_xticks(sorted(summary["scale"].unique()))
    axis.set_ylim(0, 1)
    axis.set_title("Balanced accuracy por escala")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "scale_sweep_balanced_accuracy.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
