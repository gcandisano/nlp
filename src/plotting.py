"""Utilidades de visualización."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def setup_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 100


def save_figure(fig, path: Path, dpi: int = 150):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def plot_confusion_matrix(cm, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(title)
    save_figure(fig, save_path)
