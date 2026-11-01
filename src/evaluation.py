"""Métricas e gráficos de avaliação compartilhados pelo baseline e pelo fine-tuning."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from data import LABEL_NAMES  # noqa: E402


def metrics(y_true, y_pred) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i in (0, 1)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def plot_confusion(matrix: list[list[int]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(matrix, cmap="Blues")
    names = [LABEL_NAMES[0], LABEL_NAMES[1]]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("previsto")
    ax.set_ylabel("real")
    ax.set_title(title, fontsize=10)
    top = max(max(row) for row in matrix)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                matrix[i][j],
                ha="center",
                va="center",
                color="white" if matrix[i][j] > top / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
