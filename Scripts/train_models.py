"""
RA5olver - Model Training Pipeline
=====================================
Script 4 of 6: Training All 6 Experimental Conditions

Purpose:
    Trains and evaluates all 6 experimental conditions against the
    expert-validated ground truth labels. Each condition is logged
    with identical metrics to enable direct statistical comparison
    in R (Script 6).

Research Context:
    Dissertation: "Design Science Approach to Explainable AI for Reducing
    Hallucinations in Vulnerability Management"
    Author: Ian Rivera | Colorado Technical University | 2025

Experimental Conditions:
    Condition 1: Logistic Regression  — No SHAP (Baseline)
    Condition 2: Logistic Regression  — With SHAP (XAI)
    Condition 3: Random Forest        — No SHAP (Baseline)
    Condition 4: Random Forest        — With SHAP (XAI)
    Condition 5: Support Vector Machine — No SHAP (Baseline)
    Condition 6: Support Vector Machine — With SHAP (XAI)

Hallucination Definition (Operationalized):
    A model prediction is a hallucination when it differs from the
    expert-validated ground truth label. Hallucination rate is the
    proportion of hallucinated predictions across all test records.

    hallucination_rate = (FP + FN) / total_predictions

SHAP Integration Strategy:
    SHAP values are used as a post-hoc filter. For each prediction,
    if the top SHAP feature driving a "Yes" prediction is a low-risk
    indicator (e.g., low CVSS score, low attack complexity), the
    prediction confidence is penalized and may be reclassified.
    This models the real-world use case: XAI catches unreliable
    predictions before they reach the analyst.

Metrics Captured Per Condition (for R analysis):
    - hallucination_rate
    - false_positive_rate
    - false_negative_rate
    - precision
    - recall
    - f1_score
    - accuracy
    - roc_auc
    - shap_enabled (0/1 — treatment variable for regression)
    - model_type (LR / RF / SVM — categorical for regression)

Reproducibility:
    - All models trained with fixed random_state=42
    - Identical train/test splits across all conditions
    - All results logged to structured CSV for R import
    - Model artifacts saved for SHAP Script (Script 5)

Usage:
    python train_models.py
        --x_train Data/processed/X_train_YYYYMMDD.csv
        --x_test  Data/processed/X_test_YYYYMMDD.csv
        --y_train Data/processed/y_train_YYYYMMDD.csv
        --y_test  Data/processed/y_test_YYYYMMDD.csv

Output:
    Models/trained/lr_baseline_YYYYMMDD.pkl
    Models/trained/lr_xai_YYYYMMDD.pkl
    Models/trained/rf_baseline_YYYYMMDD.pkl
    Models/trained/rf_xai_YYYYMMDD.pkl
    Models/trained/svm_baseline_YYYYMMDD.pkl
    Models/trained/svm_xai_YYYYMMDD.pkl
    Outputs/metrics/experiment_results_YYYYMMDD.csv
    Outputs/metrics/training_report_YYYYMMDD.txt

Dependencies:
    pip install pandas numpy scikit-learn shap joblib tqdm
"""

import os
import json
import logging
import argparse
import warnings
import joblib
import shap
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    # Paths
    "model_dir": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "Models", "trained"
    ),
    "metrics_dir": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "Outputs", "metrics"
    ),

    # Fixed random state — must never change between runs
    "random_state": 42,

    # SHAP filter configuration
    # Features whose high SHAP value for "Yes" prediction is suspicious
    # (i.e., low-risk features that should not drive a High priority label)
    "shap_low_risk_indicators": [
        "cvss_score_medium",       # Medium score driving High prediction
        "cvss_attack_vector_LOCAL",  # Local-only attack driving High prediction
        "cvss_attack_complexity_HIGH", # High complexity driving High prediction
        "cvss_privileges_required_HIGH", # High privileges needed
    ],

    # Confidence penalty applied when SHAP filter fires
    "shap_penalty_threshold": 0.60,  # Predictions below this confidence get reclassified

    # Model hyperparameters — documented for dissertation methodology
    "model_params": {
        "LogisticRegression": {
            "max_iter": 1000,
            "random_state": 42,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
        "RandomForest": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "class_weight": "balanced",
            "n_jobs": -1,
        },
        "SVM": {
            "kernel": "rbf",
            "probability": True,   # Required for SHAP KernelExplainer
            "random_state": 42,
            "class_weight": "balanced",
            "C": 1.0,
        },
    },
}

# ==============================================================================
# LOGGING
# ==============================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(output_dir: str, ts: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"training_log_{ts}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ==============================================================================
# MODEL DEFINITIONS
# ==============================================================================

def get_models() -> dict:
    """
    Return all 3 model instances with documented hyperparameters.
    All use fixed random_state=42 for reproducibility.
    """
    params = CONFIG["model_params"]
    return {
        "LogisticRegression": LogisticRegression(**params["LogisticRegression"]),
        "RandomForest": RandomForestClassifier(**params["RandomForest"]),
        "SVM": SVC(**params["SVM"]),
    }


# ==============================================================================
# METRICS CALCULATION
# ==============================================================================

def calculate_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       y_prob: np.ndarray,
                       model_name: str,
                       shap_enabled: int,
                       condition_id: int) -> dict:
    """
    Calculate all metrics for a single experimental condition.
    These metrics feed directly into R analysis (Script 6).

    Hallucination rate is the primary outcome variable for H0/H1 testing.

    Args:
        y_true       : Ground truth labels
        y_pred       : Model predictions
        y_prob       : Prediction probabilities (class 1)
        model_name   : "LogisticRegression" / "RandomForest" / "SVM"
        shap_enabled : 0 = baseline, 1 = XAI condition
        condition_id : 1-6 experimental condition number

    Returns:
        Dict of all metrics for this condition
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = len(y_true)

    # Primary outcome: hallucination rate
    # Hallucinations = all incorrect predictions (FP + FN)
    hallucinations = fp + fn
    hallucination_rate = hallucinations / total

    # Secondary metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = None

    return {
        # Identification
        "condition_id": condition_id,
        "model_type": model_name,
        "shap_enabled": shap_enabled,
        "timestamp": timestamp,

        # Sample counts
        "total_predictions": total,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_hallucinations": int(hallucinations),

        # Primary outcome variable (H0/H1)
        "hallucination_rate": round(hallucination_rate, 6),

        # Secondary metrics
        "false_positive_rate": round(fpr, 6),
        "false_negative_rate": round(fnr, 6),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 6),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 6),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 6),
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "roc_auc": round(auc, 6) if auc is not None else None,

        # Hallucination reduction (populated after both conditions run)
        "hallucination_reduction_vs_baseline": None,
        "hallucination_reduction_pct": None,
    }


# ==============================================================================
# SHAP INTEGRATION
# ==============================================================================

def get_shap_explainer(model, X_train: pd.DataFrame, model_name: str):
    """
    Return appropriate SHAP explainer for each model type.

    - RandomForest     -> TreeExplainer (fast, exact)
    - LogisticRegression -> LinearExplainer (fast, exact)
    - SVM              -> KernelExplainer (slower, model-agnostic)
                          Uses sample of 100 background records for speed

    Args:
        model      : Fitted sklearn model
        X_train    : Training features (for background distribution)
        model_name : Model identifier string

    Returns:
        SHAP explainer instance
    """
    logger.info(f"  Initializing SHAP explainer for {model_name}...")

    if model_name == "RandomForest":
        return shap.TreeExplainer(model)

    elif model_name == "LogisticRegression":
        return shap.LinearExplainer(model, X_train)

    elif model_name == "SVM":
        # KernelExplainer requires background dataset
        # Sample 100 records for computational feasibility
        background = shap.sample(X_train, 100, random_state=CONFIG["random_state"])
        return shap.KernelExplainer(model.predict_proba, background)

    else:
        logger.warning(f"Unknown model {model_name}. Using KernelExplainer.")
        background = shap.sample(X_train, 100, random_state=CONFIG["random_state"])
        return shap.KernelExplainer(model.predict_proba, background)


def compute_shap_values(explainer, X_test: pd.DataFrame,
                         model_name: str) -> np.ndarray:
    """
    Compute SHAP values for test set.
    Handles output format differences between explainer types.

    Returns:
        2D array of SHAP values — shape (n_samples, n_features)
        Values represent contribution to "Yes" (class 1) prediction.
    """
    logger.info(f"  Computing SHAP values for {len(X_test)} test records...")

    try:
        raw = explainer.shap_values(X_test)

        # TreeExplainer returns list [class_0_values, class_1_values]
        if isinstance(raw, list):
            return np.array(raw[1])  # Use class 1 (Priority: Yes)

        # LinearExplainer and KernelExplainer return 2D array directly
        return np.array(raw)

    except Exception as e:
        logger.error(f"SHAP computation failed for {model_name}: {e}")
        return np.zeros((len(X_test), X_test.shape[1]))


def apply_shap_filter(y_pred: np.ndarray,
                       y_prob: np.ndarray,
                       shap_values: np.ndarray,
                       feature_names: list) -> tuple:
    """
    Apply SHAP-based hallucination filter to predictions.

    Logic:
        For each "Yes" prediction (priority=1):
        1. Identify top SHAP feature driving that prediction
        2. If that feature is a known low-risk indicator AND
           prediction confidence is below threshold:
           -> Reclassify as "No" (priority=0)

        This operationalizes XAI as a hallucination filter:
        predictions driven by low-risk features with low confidence
        are flagged as likely hallucinations and corrected.

    Args:
        y_pred        : Original model predictions
        y_prob        : Prediction probabilities
        shap_values   : SHAP values array (n_samples, n_features)
        feature_names : List of feature names

    Returns:
        (filtered_predictions, filter_log)
        filter_log documents every correction for dissertation traceability
    """
    filtered = y_pred.copy()
    filter_log = []
    low_risk_set = set(CONFIG["shap_low_risk_indicators"])

    for i in range(len(y_pred)):
        # Only examine "Yes" predictions
        if y_pred[i] != 1:
            continue

        confidence = y_prob[i]

        # Find top SHAP feature for this prediction
        shap_row = shap_values[i]
        # Only consider positive SHAP values (features pushing toward "Yes")
        positive_shap = {
            feature_names[j]: shap_row[j]
            for j in range(len(feature_names))
            if shap_row[j] > 0
        }

        if not positive_shap:
            continue

        top_feature = max(positive_shap, key=positive_shap.get)
        top_shap_value = positive_shap[top_feature]

        # Apply filter: low-risk top feature + low confidence = likely hallucination
        if (top_feature in low_risk_set and
                confidence < CONFIG["shap_penalty_threshold"]):

            filtered[i] = 0  # Reclassify as No
            filter_log.append({
                "record_index": i,
                "original_prediction": 1,
                "filtered_prediction": 0,
                "confidence": round(float(confidence), 4),
                "top_shap_feature": top_feature,
                "top_shap_value": round(float(top_shap_value), 6),
                "filter_reason": (
                    f"Low-risk indicator '{top_feature}' driving Yes prediction "
                    f"with confidence {confidence:.2f} < threshold "
                    f"{CONFIG['shap_penalty_threshold']}"
                )
            })

    logger.info(
        f"  SHAP filter applied: {len(filter_log)} predictions reclassified "
        f"({len(filter_log)/len(y_pred)*100:.1f}% of test set)"
    )

    return filtered, filter_log


# ==============================================================================
# EXPERIMENT RUNNER
# ==============================================================================

def run_condition(condition_id: int,
                   model_name: str,
                   model,
                   X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_train: pd.Series,
                   y_test: pd.Series,
                   shap_enabled: bool,
                   model_dir: str) -> tuple:
    """
    Train and evaluate a single experimental condition.

    Args:
        condition_id  : 1-6
        model_name    : "LogisticRegression" / "RandomForest" / "SVM"
        model         : Untrained sklearn model instance
        X_train/test  : Feature matrices
        y_train/test  : Label vectors
        shap_enabled  : True = XAI condition; False = baseline
        model_dir     : Directory to save trained model

    Returns:
        (metrics_dict, shap_values_or_None, filter_log_or_None)
    """
    condition_label = f"Condition {condition_id}: {model_name} ({'XAI' if shap_enabled else 'Baseline'})"
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {condition_label}")
    logger.info(f"{'='*60}")

    feature_names = list(X_train.columns)

    # --- Train ---
    logger.info(f"  Training {model_name}...")
    model.fit(X_train, y_train)
    logger.info(f"  Training complete.")

    # --- Baseline Predictions ---
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    shap_values = None
    filter_log = None

    # --- SHAP Integration (XAI conditions only) ---
    if shap_enabled:
        explainer = get_shap_explainer(model, X_train, model_name)
        shap_values = compute_shap_values(explainer, X_test, model_name)
        y_pred, filter_log = apply_shap_filter(
            y_pred, y_prob, shap_values, feature_names
        )
        # Recalculate probabilities for filtered predictions
        # (keep original probs for AUC, use filtered preds for classification metrics)

    # --- Calculate Metrics ---
    metrics = calculate_metrics(
        y_true=y_test.values,
        y_pred=y_pred,
        y_prob=y_prob,
        model_name=model_name,
        shap_enabled=int(shap_enabled),
        condition_id=condition_id
    )

    # --- Save Model Artifact ---
    suffix = "xai" if shap_enabled else "baseline"
    model_key = model_name.lower().replace("logisticregression", "lr") \
                                   .replace("randomforest", "rf") \
                                   .replace("svm", "svm")
    model_path = os.path.join(model_dir, f"{model_key}_{suffix}_{timestamp}.pkl")
    joblib.dump(model, model_path)
    metrics["model_path"] = model_path
    logger.info(f"  Model saved: {model_path}")

    # --- Log Key Results ---
    logger.info(f"  Hallucination Rate : {metrics['hallucination_rate']:.4f}")
    logger.info(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
    logger.info(f"  F1 Score           : {metrics['f1_score']:.4f}")
    logger.info(f"  Accuracy           : {metrics['accuracy']:.4f}")
    if metrics["roc_auc"]:
        logger.info(f"  ROC AUC            : {metrics['roc_auc']:.4f}")

    return metrics, shap_values, filter_log


def compute_hallucination_reduction(results: list) -> list:
    """
    For each XAI condition, calculate hallucination reduction
    relative to its paired baseline condition.

    Pairs:
        Condition 1 (LR Baseline) -> Condition 2 (LR XAI)
        Condition 3 (RF Baseline) -> Condition 4 (RF XAI)
        Condition 5 (SVM Baseline)-> Condition 6 (SVM XAI)
    """
    pairs = [(1, 2), (3, 4), (5, 6)]

    results_by_id = {r["condition_id"]: r for r in results}

    for baseline_id, xai_id in pairs:
        if baseline_id in results_by_id and xai_id in results_by_id:
            baseline_rate = results_by_id[baseline_id]["hallucination_rate"]
            xai_rate = results_by_id[xai_id]["hallucination_rate"]

            reduction = baseline_rate - xai_rate
            reduction_pct = (reduction / baseline_rate * 100) if baseline_rate > 0 else 0.0

            results_by_id[xai_id]["hallucination_reduction_vs_baseline"] = round(reduction, 6)
            results_by_id[xai_id]["hallucination_reduction_pct"] = round(reduction_pct, 2)

            logger.info(
                f"\n  {results_by_id[xai_id]['model_type']} XAI vs Baseline: "
                f"Hallucination reduction = {reduction:.4f} ({reduction_pct:.1f}%)"
            )

    return list(results_by_id.values())


# ==============================================================================
# REPORTING
# ==============================================================================

def generate_training_report(results: list, ts: str) -> str:
    """
    Generate training report summarizing all 6 conditions.
    Formatted for dissertation methodology/results section reference.
    """
    header = f"""
================================================================================
RA5olver Model Training Report — All Experimental Conditions
================================================================================
Timestamp     : {ts}
Random State  : {CONFIG["random_state"]}
SHAP Penalty  : confidence < {CONFIG["shap_penalty_threshold"]}

EXPERIMENTAL CONDITIONS SUMMARY
---------------------------------
{"Cond":<6} {"Model":<22} {"SHAP":<8} {"Halluc. Rate":<14} {"FPR":<10} {"F1":<8} {"AUC":<8} {"Halluc. Reduction"}
{"-"*90}
"""
    rows = ""
    for r in sorted(results, key=lambda x: x["condition_id"]):
        reduction = (
            f"{r['hallucination_reduction_pct']:.1f}%"
            if r["hallucination_reduction_pct"] is not None
            else "— (baseline)"
        )
        auc = f"{r['roc_auc']:.4f}" if r["roc_auc"] else "N/A"
        rows += (
            f"{r['condition_id']:<6} "
            f"{r['model_type']:<22} "
            f"{'Yes' if r['shap_enabled'] else 'No':<8} "
            f"{r['hallucination_rate']:.4f}{'':8} "
            f"{r['false_positive_rate']:.4f}{'':4} "
            f"{r['f1_score']:.4f}{'':2} "
            f"{auc:<8} "
            f"{reduction}\n"
        )

    footer = f"""
INTERPRETATION NOTES
---------------------
- Hallucination Rate = (FP + FN) / Total Predictions
- Hallucination Reduction = Baseline Rate - XAI Rate (positive = improvement)
- All conditions trained on identical features (Script 3 output)
- All conditions evaluated on identical held-out test set
- Results saved to CSV for R statistical analysis (Script 6)

NEXT STEP
---------
Run: python run_experiments.py (Script 5 — SHAP visualization & deep analysis)
Then: python export_for_r.py (Script 6 — R-ready output)
================================================================================
"""
    return header + rows + footer


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="RA5olver Script 4: Train All 6 Experimental Conditions"
    )
    parser.add_argument("--x_train", required=True, help="X_train CSV from Script 3")
    parser.add_argument("--x_test",  required=True, help="X_test CSV from Script 3")
    parser.add_argument("--y_train", required=True, help="y_train CSV from Script 3")
    parser.add_argument("--y_test",  required=True, help="y_test CSV from Script 3")
    return parser.parse_args()


def main():
    global logger, timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir   = CONFIG["model_dir"]
    metrics_dir = CONFIG["metrics_dir"]

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    logger = setup_logging(metrics_dir, timestamp)

    logger.info("=" * 60)
    logger.info("RA5olver - Script 4: Model Training Pipeline")
    logger.info("=" * 60)

    args = parse_args()

    # Load feature matrices from Script 3
    logger.info("Loading train/test splits...")
    X_train = pd.read_csv(args.x_train)
    X_test  = pd.read_csv(args.x_test)
    y_train = pd.read_csv(args.y_train).squeeze()
    y_test  = pd.read_csv(args.y_test).squeeze()

    logger.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    logger.info(f"y_train distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"y_test distribution:  {y_test.value_counts().to_dict()}")

    # Define all 6 experimental conditions
    # (model_name, shap_enabled, condition_id)
    conditions = [
        (1, "LogisticRegression", False),
        (2, "LogisticRegression", True),
        (3, "RandomForest",       False),
        (4, "RandomForest",       True),
        (5, "SVM",                False),
        (6, "SVM",                True),
    ]

    all_results = []
    all_filter_logs = []

    # Run all 6 conditions
    for condition_id, model_name, shap_enabled in tqdm(
        conditions, desc="Running Experimental Conditions"
    ):
        # Fresh model instance for each condition
        model = get_models()[model_name]

        metrics, shap_values, filter_log = run_condition(
            condition_id=condition_id,
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            shap_enabled=shap_enabled,
            model_dir=model_dir
        )

        all_results.append(metrics)

        # Save SHAP filter log for XAI conditions
        if filter_log is not None:
            filter_log_file = os.path.join(
                metrics_dir,
                f"shap_filter_log_condition{condition_id}_{timestamp}.json"
            )
            with open(filter_log_file, "w") as f:
                json.dump(filter_log, f, indent=2)
            logger.info(f"  SHAP filter log saved: {filter_log_file}")
            all_filter_logs.extend(filter_log)

    # Compute hallucination reduction across pairs
    all_results = compute_hallucination_reduction(all_results)

    # Save results to CSV (primary input for R analysis)
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(metrics_dir, f"experiment_results_{timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    logger.info(f"\nExperiment results saved: {results_file}")

    # Save results to JSON (for reference)
    results_json = os.path.join(metrics_dir, f"experiment_results_{timestamp}.json")
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate and save training report
    report = generate_training_report(all_results, timestamp)
    print(report)

    report_file = os.path.join(metrics_dir, f"training_report_{timestamp}.txt")
    with open(report_file, "w") as f:
        f.write(report)

    logger.info("Script 4 complete.")
    logger.info("Next: python run_experiments.py (Script 5)")


if __name__ == "__main__":
    main()
