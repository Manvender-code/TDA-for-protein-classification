import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve
)
from src.config import RANDOM_STATE, CV_FOLDS


def train_and_evaluate(X, y):
    """
    Train RandomForest on X,y with StratifiedKFold and return:
    - metrics dict (means)
    - ROC tuple (mean_fpr, mean_tpr, mean_auc)
    - PR tuple (precision_curve, recall_curve) aggregated across folds
    - confusion tuple (all_true, final_preds) where final_preds = thresholds on all_probs
    - fitted classifier (on last fold)
    """
    clf = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    aucs = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    mccs = []

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    all_probs = []
    all_true = []

    for train_idx, test_idx in skf.split(X, y):
        clf.fit(X[train_idx], y[train_idx])

        probs = clf.predict_proba(X[test_idx])[:, 1]
        preds = (probs >= 0.5).astype(int)

        all_probs.extend(probs.tolist())
        all_true.extend(y[test_idx].tolist())

        aucs.append(roc_auc_score(y[test_idx], probs))
        accuracies.append(accuracy_score(y[test_idx], preds))
        precisions.append(precision_score(y[test_idx], preds, zero_division=0))
        recalls.append(recall_score(y[test_idx], preds, zero_division=0))
        f1s.append(f1_score(y[test_idx], preds, zero_division=0))
        mccs.append(matthews_corrcoef(y[test_idx], preds))

        fpr, tpr, _ = roc_curve(y[test_idx], probs)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    metrics = {
        "ROC_AUC_mean": float(np.mean(aucs)),
        "ROC_AUC_std": float(np.std(aucs)),
        "Accuracy_mean": float(np.mean(accuracies)),
        "Precision_mean": float(np.mean(precisions)),
        "Recall_mean": float(np.mean(recalls)),
        "F1_mean": float(np.mean(f1s)),
        "MCC_mean": float(np.mean(mccs))
    }

    mean_tpr = np.mean(tprs, axis=0)

    # Precision-Recall aggregated on pooled probabilities
    precision_curve, recall_curve, _ = precision_recall_curve(np.array(all_true), np.array(all_probs))

    final_preds = (np.array(all_probs) >= 0.5).astype(int)

    return (
        metrics,
        (mean_fpr, mean_tpr, metrics["ROC_AUC_mean"]),
        (precision_curve, recall_curve),
        (np.array(all_true), final_preds),
        clf
    )