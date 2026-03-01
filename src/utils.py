import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

from src.config import DATA_DIR, OUTPUT_DIR


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------- ROC --------------------

def plot_roc_curve(fpr, tpr, roc_auc, output_path):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_path, dpi=300)
    plt.close()


# -------------------- Precision Recall --------------------

def plot_precision_recall_curve(precision, recall, output_path):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(alpha=0.3)
    plt.savefig(output_path, dpi=300)
    plt.close()


# -------------------- Confusion Matrix --------------------

def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Enzyme", "Enzyme"],
                yticklabels=["Non-Enzyme", "Enzyme"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(output_path, dpi=300)
    plt.close()


# -------------------- Metrics Bar Plot --------------------

def plot_metrics_bar(metrics_dict, output_path):
    metrics_to_plot = {
        k: v for k, v in metrics_dict.items()
        if "mean" in k and "std" not in k
    }

    names = list(metrics_to_plot.keys())
    values = list(metrics_to_plot.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.title("Model Performance Metrics")
    plt.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 f"{height:.2f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# -------------------- Save CSV --------------------

def save_metrics(metrics, path):
    pd.DataFrame([metrics]).to_csv(path, index=False)
