import os
import warnings
warnings.filterwarnings("ignore")

from src.data_loader import download_data, load_graphs
from src.tda_pipeline import GraphToFeatures
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


def main():

    print("\n Thesis Result for protein classification problem. ")

    #setting up directories for data and output folder
    ensure_dirs() 

    # download the dataset and unzip it
    download_data()

    # graphs is array of networkx graph and labels is classification of that graph
    graphs, labels = load_graphs() 
    print(f"Total graphs in the dataset is {len(graphs)}")

    # creating feature matrix
    pipeline = GraphToFeatures.get_pipeline()
    print("\n creating numerical feature matrix(topological features + node features) for each graph of dataset")

    # passing graph to our feature matrix and we geed "X" that is numerical feature matrix
    X = pipeline.fit_transform(graphs)


    print("\nTraining & Cross-validating model...")
    metrics, roc_data, pr_data, cm_data, model = train_and_evaluate(X, labels)

    print("\n=== FINAL RESULTS (10-Fold CV) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Save metrics
    save_metrics(metrics, os.path.join(OUTPUT_DIR, "metrics.csv"))

    # ROC curve
    fpr, tpr, auc = roc_data
    plot_roc_curve(fpr, tpr, auc,
                   os.path.join(OUTPUT_DIR, "roc_curve.png"))

    # Precision-Recall
    precision_curve, recall_curve = pr_data
    plot_precision_recall_curve(
        precision_curve,
        recall_curve,
        os.path.join(OUTPUT_DIR, "precision_recall_curve.png")
    )

    # Confusion Matrix
    y_true, y_pred = cm_data
    plot_confusion_matrix(
        y_true,
        y_pred,
        os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )

    # Metrics Bar Plot
    plot_metrics_bar(
        metrics,
        os.path.join(OUTPUT_DIR, "metrics_bar_plot.png")
    )

    print("\nAll plots & metrics saved inside /output directory.")


if __name__ == "__main__":
    main()
