"""Utilidades de visualización."""

from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
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


def plot_top_terms_horizontal(
    top_fake: pd.DataFrame,
    top_real: pd.DataFrame,
    *,
    n: int = 15,
    save_path: Path | None = None,
) -> plt.Figure:
    """Barras horizontales de términos asociados a fake vs real."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panels = (
        (axes[0], top_fake.head(n), "Términos asociados a FAKE", "#e74c3c"),
        (
            axes[1],
            top_real.head(n).assign(
                coefficient=lambda frame: frame["coefficient"].abs()
            ),
            "Términos asociados a REAL",
            "#2ecc71",
        ),
    )
    for ax, data, title, color in panels:
        sns.barplot(data=data.iloc[::-1], y="term", x="coefficient", color=color, ax=ax)
        ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path)
    return fig


def plot_adjective_frequency_comparison(
    fake_counts: Counter[str],
    real_counts: Counter[str],
    *,
    top_n: int = 15,
    save_path: Path | None = None,
) -> plt.Figure:
    """Barras comparativas de adjetivos más frecuentes por clase."""
    fake_top = pd.DataFrame(
        fake_counts.most_common(top_n), columns=["adjective", "fake_count"]
    )
    real_top = pd.DataFrame(
        real_counts.most_common(top_n), columns=["adjective", "real_count"]
    )
    adj_plot = fake_top.merge(real_top, on="adjective", how="outer").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(adj_plot))
    width = 0.35
    ax.barh(
        [pos - width / 2 for pos in x],
        adj_plot["fake_count"],
        width,
        label="Fake",
        color="#e74c3c",
    )
    ax.barh(
        [pos + width / 2 for pos in x],
        adj_plot["real_count"],
        width,
        label="Real",
        color="#2ecc71",
    )
    ax.set_yticks(list(x))
    ax.set_yticklabels(adj_plot["adjective"])
    ax.set_xlabel("Frecuencia")
    ax.set_title("Adjetivos más frecuentes por clase")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path)
    return fig


def plot_wordcloud_from_frequencies(
    frequencies: Mapping[str, int],
    *,
    title: str,
    colormap: str = "Reds",
    save_path: Path | None = None,
) -> plt.Figure | None:
    """Word cloud a partir de frecuencias (p. ej. adjetivos por clase)."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        return None

    if not frequencies:
        return None

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap=colormap,
    ).generate_from_frequencies(dict(frequencies))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path)
    return fig
