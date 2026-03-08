import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

from src.data_loader import download_data, load_graphs
from src.tda_features import GraphToFeatures
from src.baseline_features import extract_baseline_features
from src.model import train_and_evaluate
from src.utils import (
    ensure_dirs,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_metrics_bar,
    save_metrics
)
from src.config import OUTPUT_DIR


def run_experiment(X, labels, folder_name):
    """
    Train and evaluate on X, labels and save plots/metrics in OUTPUT_DIR/folder_name
    """
    metrics, roc_data, pr_data, cm_data, model = train_and_evaluate(X, labels)

    folder = os.path.join(OUTPUT_DIR, folder_name)

    # Save metrics CSV
    save_metrics(metrics, os.path.join(folder, "metrics.csv"))

    # ROC
    fpr, tpr, auc = roc_data
    plot_roc_curve(fpr, tpr, auc, os.path.join(folder, "roc_curve.png"))

    # Precision-Recall
    precision_curve, recall_curve = pr_data
    plot_precision_recall_curve(precision_curve, recall_curve, os.path.join(folder, "precision_recall_curve.png"))

    # Confusion Matrix
    y_true, y_pred = cm_data
    plot_confusion_matrix(y_true, y_pred, os.path.join(folder, "confusion_matrix.png"))

    # Metrics Bar Plot
    plot_metrics_bar(metrics, os.path.join(folder, "metrics_bar_plot.png"))

    return metrics


def main():
    print("\n Thesis Result for protein classification problem.")
    # setting up directories for data and output folder
    ensure_dirs()

    # download the dataset and unzip it
    download_data()

    # graphs is array of networkx graph and labels is classification of that graph
    graphs, labels = load_graphs()
    print(f"Total graphs in the dataset is {len(graphs)}")

    # ---------------- TDA MODEL ----------------
    print("\n===============================")
    print("Running TDA Pipeline (13 TDA features + attributes)")
    print("===============================")

    pipeline = GraphToFeatures.get_pipeline()
    print("\n creating numerical feature matrix(topological features + node features) for each graph of dataset")
    X_tda = pipeline.fit_transform(graphs)
    print("TDA Feature shape:", X_tda.shape)

    tda_metrics = run_experiment(X_tda, labels, "tda")

    print("\n=== TDA FINAL RESULTS (10-Fold CV) ===")
    for k, v in tda_metrics.items():
        print(f"{k}: {v:.4f}")

    # ---------------- BASELINE MODEL ----------------
    print("\n===============================")
    print("Running Baseline Model (traditional graph features)")
    print("===============================")

    X_baseline = extract_baseline_features(graphs)
    scaler = StandardScaler()
    X_baseline = scaler.fit_transform(X_baseline)
    print("Baseline Feature shape:", X_baseline.shape)

    baseline_metrics = run_experiment(X_baseline, labels, "baseline")

    print("\n=== BASELINE FINAL RESULTS (10-Fold CV) ===")
    for k, v in baseline_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nAll plots & metrics saved inside /output/tda and /output/baseline")


if __name__ == "__main__":
    main()