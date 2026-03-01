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

    clf = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    skf = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

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

        all_probs.extend(probs)
        all_true.extend(y[test_idx])

        aucs.append(roc_auc_score(y[test_idx], probs))
        accuracies.append(accuracy_score(y[test_idx], preds))
        precisions.append(precision_score(y[test_idx], preds))
        recalls.append(recall_score(y[test_idx], preds))
        f1s.append(f1_score(y[test_idx], preds))
        mccs.append(matthews_corrcoef(y[test_idx], preds))

        fpr, tpr, _ = roc_curve(y[test_idx], probs)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    metrics = {
        "ROC_AUC_mean": np.mean(aucs),
        "ROC_AUC_std": np.std(aucs),
        "Accuracy_mean": np.mean(accuracies),
        "Precision_mean": np.mean(precisions),
        "Recall_mean": np.mean(recalls),
        "F1_mean": np.mean(f1s),
        "MCC_mean": np.mean(mccs)
    }

    mean_tpr = np.mean(tprs, axis=0)

    precision_curve, recall_curve, _ = precision_recall_curve(
        all_true, all_probs
    )

    final_preds = (np.array(all_probs) >= 0.5).astype(int)

    return (
        metrics,
        (mean_fpr, mean_tpr, metrics["ROC_AUC_mean"]),
        (precision_curve, recall_curve),
        (np.array(all_true), final_preds),
        clf
    )
