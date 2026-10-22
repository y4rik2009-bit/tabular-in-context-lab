"""Publication-ready chart from the frozen three-seed benchmark."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    path = Path("results/verified/metrics.csv")
    if not path.exists():
        raise SystemExit("Run the benchmark and freeze results first")
    df = pd.read_csv(path)
    methods = ["MiniTabICL", "5-NN", "Logistic regression", "CatBoost"]
    colors = ["#087e8b", "#a7bcc0", "#b47437", "#344654"]
    datasets = list(dict.fromkeys(df["dataset"]))
    means = df.groupby(["dataset", "method"])[["auc", "log_loss"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
    x = np.arange(len(datasets))
    for j, (method, color) in enumerate(zip(methods, colors)):
        xx = x + (j - 1.5) * 0.19
        axes[0].bar(xx, [means.loc[(name, method), "auc"] for name in datasets],
                    width=0.18, color=color, label=method)
        axes[1].bar(xx, [means.loc[(name, method), "log_loss"] for name in datasets],
                    width=0.18, color=color)
    for ax in axes:
        ax.set_xticks(x, ["Banknote", "Breast cancer", "Ionosphere"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].set(title="ROC AUC (higher is better)", ylim=(0, 1.0))
    axes[1].set(title="Log loss (lower is better)", ylim=(0, None))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/benchmark.svg", bbox_inches="tight")
    fig.savefig("figures/benchmark.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    ablations = ["MiniTabICL", "MiniTabICL, random weights",
                 "MiniTabICL, shuffled context labels"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for j, (method, color) in enumerate(zip(ablations, colors[:3])):
        ax.bar(x + (j - 1) * 0.25,
               [means.loc[(name, method), "auc"] for name in datasets],
               width=0.23, color=color, label=method)
    ax.set_xticks(x, ["Banknote", "Breast cancer", "Ionosphere"])
    ax.set(title="Context and pretraining ablations · ROC AUC", ylim=(0, 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    fig.savefig("figures/ablations.svg", bbox_inches="tight")
    fig.savefig("figures/ablations.png", dpi=180, bbox_inches="tight")
    print("Saved figures/benchmark.svg and figures/ablations.svg")


if __name__ == "__main__":
    main()
