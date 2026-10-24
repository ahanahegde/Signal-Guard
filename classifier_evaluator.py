"""
SignalGuard Baseline Classifier Evaluation Module.

Computes comprehensive classification metrics:
- Confusion Matrix (TN, FP, FN, TP)
- Precision, Recall (Sensitivity), Specificity
- False Positive Rate (FPR), False Negative Rate (FNR)
- F1-Score, Balanced Accuracy, and ROC-AUC
- Per-condition breakdown (e.g. compression levels, manipulation techniques)

Strictly separates evaluation splits and reports transparent, uninflated metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from baseline_classifier import BaselineClassifier
from dataset_loader import (
    LABEL_GENUINE,
    LABEL_MANIPULATED,
    DatasetLoader,
    VideoRecord,
)


@dataclass
class EvaluationMetrics:
    split_name: str
    total_samples: int
    n_genuine: int
    n_manipulated: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float
    f1_score: float
    balanced_accuracy: float
    roc_auc: float | None
    per_condition: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics_from_arrays(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float] | None = None,
    split_name: str = "evaluation",
) -> dict[str, Any]:
    """
    Compute binary classification metrics from ground truth and predictions.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    total = len(y_t)
    if total == 0:
        return {"error": "Empty sample array provided."}

    n_manip = int(np.sum(y_t == LABEL_MANIPULATED))
    n_gen = int(np.sum(y_t == LABEL_GENUINE))

    # Confusion matrix with explicit labels [0, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        cm = confusion_matrix(y_t, y_p, labels=[LABEL_GENUINE, LABEL_MANIPULATED])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(f1_score(y_t, y_p, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_t, y_p))

    auc = None
    if y_prob is not None and len(np.unique(y_t)) > 1:
        try:
            auc = float(roc_auc_score(y_t, y_prob))
        except Exception:
            auc = None

    return {
        "split_name": split_name,
        "total_samples": total,
        "n_genuine": n_gen,
        "n_manipulated": n_manip,
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "matrix_2x2": [[tn, fp], [fn, tp]],
        },
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
    }


def evaluate_records(
    classifier: BaselineClassifier,
    records: Sequence[VideoRecord],
    split_name: str = "test",
) -> dict[str, Any]:
    """
    Run inference and evaluate a classifier over a list of VideoRecords.
    Includes overall and per-condition evaluation.
    """
    if not records:
        return {"error": "No records to evaluate."}

    y_true = []
    y_pred = []
    y_prob = []
    condition_data: dict[str, dict[str, list[Any]]] = {}
    skipped = 0

    for r in records:
        res = classifier.predict_single(r.video_path)
        if res["status"] != "SUCCESS" or res["probability_manipulated"] is None:
            skipped += 1
            continue

        pred_label = LABEL_MANIPULATED if res["prediction"] == "MANIPULATED" else LABEL_GENUINE
        prob_manip = float(res["probability_manipulated"])

        y_true.append(r.label)
        y_pred.append(pred_label)
        y_prob.append(prob_manip)

        cond = r.condition or "default"
        if cond not in condition_data:
            condition_data[cond] = {"y_true": [], "y_pred": [], "y_prob": []}
        condition_data[cond]["y_true"].append(r.label)
        condition_data[cond]["y_pred"].append(pred_label)
        condition_data[cond]["y_prob"].append(prob_manip)

    if not y_true:
        return {"error": "All video records failed during feature extraction / prediction."}

    metrics = compute_metrics_from_arrays(y_true, y_pred, y_prob, split_name=split_name)
    metrics["skipped_count"] = skipped

    # Per-condition breakdown
    per_condition_metrics = {}
    for cond, data in condition_data.items():
        if data["y_true"]:
            cond_res = compute_metrics_from_arrays(
                data["y_true"],
                data["y_pred"],
                data["y_prob"],
                split_name=f"{split_name}_{cond}",
            )
            per_condition_metrics[cond] = cond_res

    metrics["per_condition"] = per_condition_metrics
    return metrics


def format_evaluation_report(metrics: dict[str, Any]) -> str:
    """Format evaluation dictionary into a clean markdown report."""
    if "error" in metrics:
        return f"Evaluation Error: {metrics['error']}"

    cm = metrics["confusion_matrix"]
    auc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else "N/A"
    lines = [
        "=" * 68,
        f"SIGNALGUARD BASELINE CLASSIFIER EVALUATION: {metrics['split_name'].upper()}",
        "=" * 68,
        f"Total evaluated samples: {metrics['total_samples']} (Skipped unreadable: {metrics.get('skipped_count', 0)})",
        f"  - Genuine samples:     {metrics['n_genuine']}",
        f"  - Manipulated samples: {metrics['n_manipulated']}",
        "",
        "Confusion Matrix:",
        f"                 Predicted Genuine    Predicted Manipulated",
        f"  Actual Genuine     TN: {cm['true_negatives']:4d}             FP: {cm['false_positives']:4d}",
        f"  Actual Manip.      FN: {cm['false_negatives']:4d}             TP: {cm['true_positives']:4d}",
        "",
        "Key Performance Metrics:",
        f"  Balanced Accuracy:     {metrics['balanced_accuracy'] * 100:6.2f}%",
        f"  F1 Score:              {metrics['f1_score'] * 100:6.2f}%",
        f"  Precision:             {metrics['precision'] * 100:6.2f}%",
        f"  Recall (Sensitivity):  {metrics['recall'] * 100:6.2f}%",
        f"  Specificity:           {metrics['specificity'] * 100:6.2f}%",
        f"  False Positive Rate:   {metrics['false_positive_rate'] * 100:6.2f}%",
        f"  False Negative Rate:   {metrics['false_negative_rate'] * 100:6.2f}%",
        f"  ROC-AUC:               {auc_str}",
    ]

    per_cond = metrics.get("per_condition", {})
    if len(per_cond) > 1:
        lines.append("")
        lines.append("Per-Condition Performance Breakdown:")
        lines.append(f"  {'Condition':<24} {'Samples':<8} {'Bal.Acc':<10} {'F1':<8} {'FPR':<8} {'FNR':<8}")
        lines.append("  " + "-" * 64)
        for cond, c_met in per_cond.items():
            lines.append(
                f"  {cond:<24} {c_met['total_samples']:<8d} "
                f"{c_met['balanced_accuracy']*100:6.2f}%   "
                f"{c_met['f1_score']*100:6.2f}%  "
                f"{c_met['false_positive_rate']*100:6.2f}%  "
                f"{c_met['false_negative_rate']*100:6.2f}%"
            )

    lines.append("=" * 68)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="SignalGuard Deepfake Classifier Evaluator")
    parser.add_argument("--model", type=str, required=True, help="Path to trained baseline model .joblib")
    parser.add_argument("--manifest", type=str, required=True, help="Path to dataset manifest CSV/JSON")
    parser.add_argument("--split", type=str, default="test", help="Which split to evaluate ('test', 'val', 'train', 'all')")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output path for JSON metrics")

    args = parser.parse_args()

    classifier = BaselineClassifier.load(args.model)
    records = DatasetLoader.load_from_manifest(args.manifest)

    if args.split.lower() != "all":
        records = [r for r in records if r.split == args.split.lower()]

    if not records:
        print(f"ERROR: No records matching split '{args.split}' found in {args.manifest}")
        return 1

    metrics = evaluate_records(classifier, records, split_name=args.split)
    report = format_evaluation_report(metrics)
    print(report)

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {out_p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
