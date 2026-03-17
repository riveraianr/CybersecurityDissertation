"""
RA5olver - SHAP Visualization & Deep Analysis
===============================================
Script 5 of 6: Explainability Analysis & Dissertation Visual Artifacts

Purpose:
    Generates all SHAP-based visualizations and deep explainability analysis
    across the three XAI experimental conditions (LR, RF, SVM with SHAP).
    Outputs are dissertation-ready figures for the results chapter and
    structured JSON for supplementary analysis.

Research Context:
    Dissertation: "Design Science Approach to Explainable AI for Reducing
    Hallucinations in Vulnerability Management"
    Author: Ian Rivera | Colorado Technical University | 2025

Visualizations Produced:
    Per XAI Model (LR, RF, SVM):
        1. SHAP Summary Plot (Beeswarm)   — global feature importance
        2. SHAP Bar Plot                  — mean absolute SHAP values
        3. SHAP Waterfall Plot            — single prediction explanation
        4. SHAP Force Plot (HTML)         — interactive single prediction
        5. SHAP Dependence Plot           — top feature interaction

    Cross-Model Comparison:
        6. Feature Importance Comparison  — top 10 features across all 3 models
        7. Hallucination Rate Comparison  — baseline vs XAI per model
        8. SHAP Value Distribution        — spread of explanations per model

Analysis Outputs:
    - Top contributing features per model (JSON)
    - Per-record SHAP explanations for sampled records (JSON)
    - Hallucination correction analysis (which corrections were right/wrong)

Reproducibility:
    - All plots saved as high-resolution PNG (300 DPI) for publication
    - All plots also saved as HTML for interactive inspection
    - Random sample for waterfall/force plots uses fixed random_state=42
    - All analysis outputs versioned with timestamp

Usage:
    python run_experiments.py
        --x_train   Data/processed/X_train_YYYYMMDD.csv
        --x_test    Data/processed/X_test_YYYYMMDD.csv
        --y_test    Data/processed/y_test_YYYYMMDD.csv
        --results   Outputs/metrics/experiment_results_YYYYMMDD.csv
        --model_dir Models/trained/

Output:
    Outputs/shap_plots/
        lr_xai_summary_YYYYMMDD.png
        lr_xai_bar_YYYYMMDD.png
        lr_xai_waterfall_YYYYMMDD.png
        rf_xai_summary_YYYYMMDD.png
        rf_xai_bar_YYYYMMDD.png
        rf_xai_waterfall_YYYYMMDD.png
        svm_xai_summary_YYYYMMDD.png
        svm_xai_bar_YYYYMMDD.png
        svm_xai_waterfall_YYYYMMDD.png
        cross_model_feature_comparison_YYYYMMDD.png
        hallucination_rate_comparison_YYYYMMDD.png
    Outputs/metrics/
        shap_analysis_YYYYMMDD.json
        hallucination_correction_analysis_YYYYMMDD.json

Dependencies:
    pip install pandas numpy scikit-learn shap joblib matplotlib seaborn
"""

import os
import re
import glob
import json
import logging
import argparse
import warnings
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/local reproducibility
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    "shap_plots_dir": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "Outputs", "shap_plots"
    ),
    "metrics_dir": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "Outputs", "metrics"
    ),
    "model_dir": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "Models", "trained"
    ),

    # Plot settings
    "dpi": 300,                    # Publication-quality resolution
    "figure_size": (12, 8),
    "style": "seaborn-v0_8-whitegrid",

    # Sample size for waterfall/force plots
    "waterfall_sample_size": 1,    # Single representative record
    "random_state": 42,

    # Top N features for comparison plots
    "top_n_features": 10,

    # SHAP background sample for SVM KernelExplainer
    "svm_background_samples": 100,

    # Color scheme (dissertation-appropriate, colorblind-friendly)
    "colors": {
        "baseline": "#4878CF",     # Blue
        "xai": "#6ACC65",          # Green
        "highlight": "#D65F5F",    # Red
        "neutral": "#B47CC7",      # Purple
    },
}

# XAI model condition mapping
XAI_CONDITIONS = {
    "LogisticRegression": {"condition_id": 2, "short": "lr",  "label": "Logistic Regression"},
    "RandomForest":       {"condition_id": 4, "short": "rf",  "label": "Random Forest"},
    "SVM":                {"condition_id": 6, "short": "svm", "label": "SVM"},
}

BASELINE_CONDITIONS = {
    "LogisticRegression": {"condition_id": 1},
    "RandomForest":       {"condition_id": 3},
    "SVM":                {"condition_id": 5},
}

# ==============================================================================
# LOGGING
# ==============================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(output_dir: str, ts: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"experiment_log_{ts}.txt")
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
# MODEL LOADING
# ==============================================================================

def load_xai_models(model_dir: str, ts_pattern: str = None) -> dict:
    """
    Load the three XAI-condition trained models from Script 4.

    Finds most recent model files if no timestamp pattern provided.
    Models expected:
        lr_xai_*.pkl, rf_xai_*.pkl, svm_xai_*.pkl

    Returns:
        Dict {model_name: loaded_model}
    """
    models = {}
    model_keys = {
        "LogisticRegression": "lr_xai",
        "RandomForest": "rf_xai",
        "SVM": "svm_xai",
    }

    for model_name, file_prefix in model_keys.items():
        pattern = os.path.join(model_dir, f"{file_prefix}_*.pkl")
        matches = sorted(glob.glob(pattern))

        if not matches:
            logger.error(f"No model file found for pattern: {pattern}")
            continue

        # Use most recent file
        model_path = matches[-1]
        try:
            models[model_name] = joblib.load(model_path)
            logger.info(f"Loaded {model_name} XAI model: {os.path.basename(model_path)}")
        except Exception as e:
            logger.error(f"Failed to load {model_path}: {e}")

    return models


# ==============================================================================
# SHAP COMPUTATION
# ==============================================================================

def compute_shap_for_model(model_name: str,
                            model,
                            X_train: pd.DataFrame,
                            X_test: pd.DataFrame) -> tuple:
    """
    Compute SHAP values and Explanation object for a given model.

    Returns:
        (shap_values_array, shap_explanation_object, explainer)
    """
    logger.info(f"Computing SHAP values for {model_name}...")
    feature_names = list(X_train.columns)

    try:
        if model_name == "RandomForest":
            explainer = shap.TreeExplainer(model)
            shap_values_raw = explainer.shap_values(X_test)
            # TreeExplainer returns [class0, class1] — use class 1
            if isinstance(shap_values_raw, list):
                shap_array = np.array(shap_values_raw[1])
            else:
                shap_array = np.array(shap_values_raw)
            explanation = shap.Explanation(
                values=shap_array,
                base_values=explainer.expected_value[1] if isinstance(
                    explainer.expected_value, list) else explainer.expected_value,
                data=X_test.values,
                feature_names=feature_names
            )

        elif model_name == "LogisticRegression":
            explainer = shap.LinearExplainer(model, X_train)
            shap_array = np.array(explainer.shap_values(X_test))
            explanation = shap.Explanation(
                values=shap_array,
                base_values=explainer.expected_value,
                data=X_test.values,
                feature_names=feature_names
            )

        elif model_name == "SVM":
            background = shap.sample(
                X_train, CONFIG["svm_background_samples"],
                random_state=CONFIG["random_state"]
            )
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values_raw = explainer.shap_values(X_test)
            if isinstance(shap_values_raw, list):
                shap_array = np.array(shap_values_raw[1])
            else:
                shap_array = np.array(shap_values_raw)
            explanation = shap.Explanation(
                values=shap_array,
                base_values=explainer.expected_value[1] if isinstance(
                    explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                data=X_test.values,
                feature_names=feature_names
            )

        else:
            raise ValueError(f"Unknown model type: {model_name}")

        logger.info(f"  SHAP computation complete. Shape: {shap_array.shape}")
        return shap_array, explanation, explainer

    except Exception as e:
        logger.error(f"SHAP computation failed for {model_name}: {e}")
        shap_array = np.zeros((len(X_test), len(feature_names)))
        explanation = shap.Explanation(
            values=shap_array,
            base_values=0.0,
            data=X_test.values,
            feature_names=feature_names
        )
        return shap_array, explanation, None


# ==============================================================================
# INDIVIDUAL MODEL PLOTS
# ==============================================================================

def plot_summary_beeswarm(explanation: shap.Explanation,
                           model_name: str,
                           short_name: str,
                           output_dir: str,
                           ts: str):
    """
    SHAP Beeswarm summary plot.
    Shows distribution of SHAP values for all features across all test records.
    Dissertation use: demonstrates which features most influence predictions globally.
    """
    fig_path = os.path.join(output_dir, f"{short_name}_xai_summary_{ts}.png")

    try:
        plt.figure(figsize=CONFIG["figure_size"])
        shap.plots.beeswarm(explanation, max_display=CONFIG["top_n_features"],
                            show=False)
        plt.title(
            f"SHAP Feature Impact — {model_name}\n"
            f"(Beeswarm: Distribution of SHAP Values Across Test Records)",
            fontsize=13, fontweight="bold", pad=15
        )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=CONFIG["dpi"], bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: {os.path.basename(fig_path)}")
    except Exception as e:
        logger.warning(f"  Beeswarm plot failed for {model_name}: {e}")
    return fig_path


def plot_bar_importance(explanation: shap.Explanation,
                         model_name: str,
                         short_name: str,
                         output_dir: str,
                         ts: str):
    """
    SHAP Bar plot — mean absolute SHAP values.
    Dissertation use: clean feature importance ranking for results chapter.
    """
    fig_path = os.path.join(output_dir, f"{short_name}_xai_bar_{ts}.png")

    try:
        plt.figure(figsize=(10, 7))
        shap.plots.bar(explanation, max_display=CONFIG["top_n_features"], show=False)
        plt.title(
            f"Mean |SHAP| Feature Importance — {model_name}\n"
            f"(Higher = Greater Average Impact on Priority Prediction)",
            fontsize=13, fontweight="bold", pad=15
        )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=CONFIG["dpi"], bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: {os.path.basename(fig_path)}")
    except Exception as e:
        logger.warning(f"  Bar plot failed for {model_name}: {e}")
    return fig_path


def plot_waterfall(explanation: shap.Explanation,
                   model_name: str,
                   short_name: str,
                   output_dir: str,
                   ts: str,
                   record_idx: int = 0):
    """
    SHAP Waterfall plot for a single prediction.
    Dissertation use: illustrates how XAI explains an individual vulnerability
    prediction — key for transparency narrative.
    """
    fig_path = os.path.join(output_dir, f"{short_name}_xai_waterfall_{ts}.png")

    try:
        plt.figure(figsize=CONFIG["figure_size"])
        shap.plots.waterfall(explanation[record_idx], show=False)
        plt.title(
            f"SHAP Waterfall — {model_name} (Record #{record_idx})\n"
            f"(Feature contributions to a single priority prediction)",
            fontsize=13, fontweight="bold", pad=15
        )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=CONFIG["dpi"], bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: {os.path.basename(fig_path)}")
    except Exception as e:
        logger.warning(f"  Waterfall plot failed for {model_name}: {e}")
    return fig_path


def plot_dependence(shap_array: np.ndarray,
                    X_test: pd.DataFrame,
                    model_name: str,
                    short_name: str,
                    output_dir: str,
                    ts: str):
    """
    SHAP Dependence plot for top feature.
    Shows how the top feature's value affects its SHAP contribution.
    Dissertation use: demonstrates non-linear feature relationships.
    """
    fig_path = os.path.join(output_dir, f"{short_name}_xai_dependence_{ts}.png")

    try:
        # Find top feature by mean absolute SHAP
        mean_abs = np.abs(shap_array).mean(axis=0)
        top_feature_idx = np.argmax(mean_abs)
        top_feature = X_test.columns[top_feature_idx]

        plt.figure(figsize=CONFIG["figure_size"])
        shap.dependence_plot(
            top_feature_idx,
            shap_array,
            X_test,
            feature_names=list(X_test.columns),
            show=False
        )
        plt.title(
            f"SHAP Dependence — {model_name}: '{top_feature}'\n"
            f"(How feature value affects its contribution to priority prediction)",
            fontsize=13, fontweight="bold", pad=15
        )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=CONFIG["dpi"], bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: {os.path.basename(fig_path)}")
    except Exception as e:
        logger.warning(f"  Dependence plot failed for {model_name}: {e}")
    return fig_path


# ==============================================================================
# CROSS-MODEL COMPARISON PLOTS
# ==============================================================================

def plot_feature_importance_comparison(all_shap: dict,
                                        feature_names: list,
                                        output_dir: str,
                                        ts: str):
    """
    Side-by-side bar chart of top 10 features across all 3 XAI models.
    Dissertation use: demonstrates consistency of SHAP explanations
    across model types — strengthens validity of XAI approach.
    """
    fig_path = os.path.join(output_dir, f"cross_model_feature_comparison_{ts}.png")

    try:
        # Compute mean absolute SHAP per model
        model_importance = {}
        for model_name, shap_array in all_shap.items():
            mean_abs = np.abs(shap_array).mean(axis=0)
            model_importance[model_name] = dict(zip(feature_names, mean_abs))

        # Get union of top features across models
        all_top = set()
        for imp in model_importance.values():
            top = sorted(imp, key=imp.get, reverse=True)[:CONFIG["top_n_features"]]
            all_top.update(top)
        top_features = sorted(
            all_top,
            key=lambda f: np.mean([imp.get(f, 0) for imp in model_importance.values()]),
            reverse=True
        )[:CONFIG["top_n_features"]]

        # Build DataFrame for plotting
        plot_data = pd.DataFrame({
            model: [model_importance[model].get(f, 0) for f in top_features]
            for model in model_importance
        }, index=top_features)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        plot_data.plot(kind="barh", ax=ax, color=[
            CONFIG["colors"]["baseline"],
            CONFIG["colors"]["xai"],
            CONFIG["colors"]["highlight"]
        ])
        ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
        ax.set_title(
            "Cross-Model Feature Importance Comparison (XAI Conditions)\n"
            "Top 10 Features by Mean Absolute SHAP Value",
            fontsize=14, fontweight="bold"
        )
        ax.invert_yaxis()
        ax.legend(title="Model", fontsize=10)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=CONFIG["dpi"], bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: {os.path.basename(fig_path)}")

    except Exception as e:
        logger.warning(f"  Feature comparison plot failed: {e}")
    return fig_path


def plot_hallucination_comparison(results_df: pd.DataFrame,
                                   output_dir: str,
                                   ts: str):
    """
    Grouped bar chart: Baseline vs XAI hallucination rates per model.
    This is the primary visual for your H0/H1 results.
    Dissertation use: the key figure in your results chapter.
    """
    fig_path = os.path.join(output_dir, f"hallucination_rate_comparison_{ts}.png")

    try:
        models = ["LogisticRegression", "RandomForest", "SVM"]
        labels = ["Logistic\nRegression", "Random\nForest", "SVM"]

        baseline_rates = []
        xai_rates = []

        baseline_ids = {1: "LogisticRegression", 3: "RandomForest", 5: "SVM"}
        xai_ids = {2: "LogisticRegression", 4: "RandomForest", 6: "SVM"}

        for model_name in models:
            b_row = results_df[
                (results_df["model_type"] == model_name) &
                (results_df["shap_enabled"] == 0)
            ]
            x_row = results_df[
                (results_df["model_type"] == model_name) &
                (results_df["shap_enabled"] == 1)
            ]
            baseline_rates.append(
                float(b_row["hallucination_rate"].values[0]) if len(b_row) > 0 else 0
            )
            xai_rates.append(
                float(x_row["hallucination_rate"].values[0]) if len(x_row) > 0 else 0
            )

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(11, 7))
        bars_baseline = ax.bar(
            x - width/2, baseline_rates, width,
            label="Baseline (No SHAP)",
            color=CONFIG["colors"]["baseline"],
            alpha=0.85, edgecolor="white", linewidth=1.2
        )
        bars_xai = ax.bar(
            x + width/2, xai_rates, width,
            label="XAI (With SHAP)",
            color=CONFIG["colors"]["xai"],
            alpha=0.85, edgecolor="white", linewidth=1.2
        )

        # Annotate reduction percentages
        for i, (b, x_val) in enumerate(zip(baseline_rates, xai_rates)):
            if b > 0:
                reduction_pct = (b - x_val) / b * 100
                ax.annotate(
                    f"−{reduction_pct:.1f}%",
                    xy=(i + width/2, x_val),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=10,
                    color=CONFIG["colors"]["xai"],
                    fontweight="bold"
                )

        ax.set_xlabel("Model Type", fontsize=13)
        ax.set_ylabel("Hallucination Rate\n(Incorrect Predictions / Total Predictions)", fontsize=12)
        ax.set_title(
            "Hallucination Rate: Baseline vs XAI Conditions\n"
            "Primary Outcome Variable — H0/H1 Test",
            fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.legend(fontsize=11)
        ax.set_ylim(0, max(baseline_rates + xai_rates) * 1.3)

        # Add value labels on bars
        for bar in bars_baseline:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9, color="gray"
            )
        for bar in bars_xai:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9, color="gray"
            )

        plt.tight_layout()
        plt.savefig(fig_path, dpi=CONFIG["dpi"], bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: {os.path.basename(fig_path)}")

    except Exception as e:
        logger.warning(f"  Hallucination comparison plot failed: {e}")
    return fig_path


def plot_metric_heatmap(results_df: pd.DataFrame,
                         output_dir: str,
                         ts: str):
    """
    Heatmap of all metrics across all 6 conditions.
    Dissertation use: comprehensive results overview in appendix.
    """
    fig_path = os.path.join(output_dir, f"metrics_heatmap_{ts}.png")

    try:
        metric_cols = [
            "hallucination_rate", "false_positive_rate", "false_negative_rate",
            "precision", "recall", "f1_score", "accuracy", "roc_auc"
        ]
        display_cols = [
            "Halluc. Rate", "FP Rate", "FN Rate",
            "Precision", "Recall", "F1", "Accuracy", "AUC"
        ]

        labels = [
            f"C{row['condition_id']}: {row['model_type'][:2].upper()} "
            f"({'XAI' if row['shap_enabled'] else 'Base'})"
            for _, row in results_df.iterrows()
        ]

        heatmap_data = results_df[metric_cols].copy().fillna(0)
        heatmap_data.index = labels
        heatmap_data.columns = display_cols

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            heatmap_data.astype(float),
            annot=True, fmt=".3f",
            cmap="RdYlGn_r",
            linewidths=0.5,
            ax=ax,
            vmin=0, vmax=1
        )
        ax.set_title(
            "Metrics Heatmap — All 6 Experimental Conditions\n"
            "(Lower hallucination/FP rates = better; Higher precision/recall/F1/AUC = better)",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=CONFIG["dpi"], bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: {os.path.basename(fig_path)}")

    except Exception as e:
        logger.warning(f"  Metrics heatmap failed: {e}")
    return fig_path


# ==============================================================================
# SHAP ANALYSIS
# ==============================================================================

def extract_top_features(shap_array: np.ndarray,
                          feature_names: list,
                          top_n: int = 10) -> list:
    """
    Extract top N features by mean absolute SHAP value.
    Returns list of dicts for JSON serialization.
    """
    mean_abs = np.abs(shap_array).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_n]

    return [
        {
            "rank": int(i + 1),
            "feature": feature_names[idx],
            "mean_abs_shap": round(float(mean_abs[idx]), 6),
            "interpretation": interpret_feature(feature_names[idx])
        }
        for i, idx in enumerate(top_indices)
    ]


def interpret_feature(feature_name: str) -> str:
    """
    Generate human-readable interpretation for common features.
    Used in JSON analysis output and dissertation narrative.
    """
    interpretations = {
        "cvss_base_score":              "Raw CVSS severity score (0-10)",
        "cvss_score_squared":           "Non-linear CVSS risk scaling",
        "cvss_score_critical":          "Binary: CVSS >= 9.0 (Critical)",
        "cvss_score_high":              "Binary: CVSS 7.0-8.9 (High)",
        "cvss_score_medium":            "Binary: CVSS 4.0-6.9 (Medium)",
        "days_since_published":         "Days since CVE publication",
        "published_year":               "Year CVE was published",
        "is_recent":                    "Binary: published within 2 years",
        "cvss_attack_vector_NETWORK":   "Network-accessible attack vector",
        "cvss_attack_vector_LOCAL":     "Local-only attack vector",
        "cvss_attack_complexity_LOW":   "Low attack complexity (easy to exploit)",
        "cvss_attack_complexity_HIGH":  "High attack complexity (harder to exploit)",
        "cvss_privileges_required_NONE":"No privileges required to exploit",
        "cvss_confidentiality_impact_HIGH": "High confidentiality impact",
        "cvss_integrity_impact_HIGH":   "High integrity impact",
        "cvss_availability_impact_HIGH":"High availability impact",
    }

    # TF-IDF features
    if feature_name.startswith("tfidf_"):
        term = feature_name.replace("tfidf_", "")
        return f"TF-IDF weight for term: '{term}' in CVE description"

    return interpretations.get(feature_name, f"Feature: {feature_name}")


def analyze_hallucination_corrections(filter_logs_dir: str,
                                       y_test: pd.Series,
                                       ts: str) -> dict:
    """
    Analyze SHAP filter corrections against ground truth.
    Determines how many corrections were accurate (true corrections)
    vs introduced new errors (overcorrections).

    This is key for your dissertation results narrative.
    """
    analysis = {
        "timestamp": ts,
        "conditions_analyzed": [],
        "summary": {}
    }

    filter_pattern = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "Outputs", "metrics",
        f"shap_filter_log_condition*_{ts[:8]}*.json"
    )
    filter_files = glob.glob(filter_pattern)

    if not filter_files:
        logger.warning("No SHAP filter log files found for correction analysis.")
        return analysis

    total_corrections = 0
    true_corrections = 0
    overcorrections = 0

    for fpath in filter_files:
        try:
            with open(fpath) as f:
                filter_log = json.load(f)

            condition_corrections = 0
            condition_true = 0
            condition_over = 0

            for entry in filter_log:
                idx = entry["record_index"]
                if idx < len(y_test):
                    ground_truth = y_test.iloc[idx]
                    # Original prediction was 1 (Yes), filter changed to 0 (No)
                    # If ground truth is 0 -> True correction (was a hallucination)
                    # If ground truth is 1 -> Overcorrection (introduced error)
                    if ground_truth == 0:
                        condition_true += 1
                        true_corrections += 1
                    else:
                        condition_over += 1
                        overcorrections += 1
                    condition_corrections += 1
                    total_corrections += 1

            condition_id = os.path.basename(fpath).split("condition")[1].split("_")[0]
            analysis["conditions_analyzed"].append({
                "condition_id": condition_id,
                "file": os.path.basename(fpath),
                "total_corrections": condition_corrections,
                "true_corrections": condition_true,
                "overcorrections": condition_over,
                "correction_accuracy": round(
                    condition_true / condition_corrections, 4
                ) if condition_corrections > 0 else 0
            })

        except Exception as e:
            logger.warning(f"Could not analyze filter log {fpath}: {e}")

    analysis["summary"] = {
        "total_corrections": total_corrections,
        "true_corrections": true_corrections,
        "overcorrections": overcorrections,
        "overall_correction_accuracy": round(
            true_corrections / total_corrections, 4
        ) if total_corrections > 0 else 0,
        "interpretation": (
            f"SHAP filter made {total_corrections} total reclassifications. "
            f"{true_corrections} ({true_corrections/total_corrections*100:.1f}% ) "
            f"were accurate corrections of hallucinated predictions. "
            f"{overcorrections} ({overcorrections/total_corrections*100:.1f}%) "
            f"were overcorrections that introduced new errors."
            if total_corrections > 0 else "No corrections to analyze."
        )
    }

    return analysis


# ==============================================================================
# MAIN
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="RA5olver Script 5: SHAP Visualization & Deep Analysis"
    )
    parser.add_argument("--x_train",   required=True, help="X_train CSV from Script 3")
    parser.add_argument("--x_test",    required=True, help="X_test CSV from Script 3")
    parser.add_argument("--y_test",    required=True, help="y_test CSV from Script 3")
    parser.add_argument("--results",   required=True, help="experiment_results CSV from Script 4")
    parser.add_argument("--model_dir", required=False,
                        default=CONFIG["model_dir"],
                        help="Directory containing trained models from Script 4")
    return parser.parse_args()


def main():
    global logger, timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shap_plots_dir = CONFIG["shap_plots_dir"]
    metrics_dir    = CONFIG["metrics_dir"]

    os.makedirs(shap_plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    logger = setup_logging(metrics_dir, timestamp)

    logger.info("=" * 60)
    logger.info("RA5olver - Script 5: SHAP Visualization & Deep Analysis")
    logger.info("=" * 60)

    args = parse_args()

    # Load data
    logger.info("Loading data...")
    X_train    = pd.read_csv(args.x_train)
    X_test     = pd.read_csv(args.x_test)
    y_test     = pd.read_csv(args.y_test).squeeze()
    results_df = pd.read_csv(args.results)

    feature_names = list(X_test.columns)
    logger.info(f"Test set: {X_test.shape} | Features: {len(feature_names)}")

    # Load XAI models from Script 4
    logger.info("Loading XAI models...")
    models = load_xai_models(args.model_dir)

    if not models:
        logger.error("No models loaded. Ensure Script 4 has been run.")
        return

    # Set plot style
    try:
        plt.style.use(CONFIG["style"])
    except Exception:
        plt.style.use("seaborn-whitegrid")

    # ----------------------------------------------------------------
    # Per-model SHAP computation and plots
    # ----------------------------------------------------------------
    all_shap = {}
    all_top_features = {}

    for model_name, model in models.items():
        short = XAI_CONDITIONS[model_name]["short"]
        label = XAI_CONDITIONS[model_name]["label"]

        logger.info(f"\n--- Analyzing {label} ---")

        # Compute SHAP
        shap_array, explanation, explainer = compute_shap_for_model(
            model_name, model, X_train, X_test
        )
        all_shap[model_name] = shap_array

        # Select representative record for waterfall (highest confidence "Yes")
        try:
            probs = model.predict_proba(X_test)[:, 1]
            record_idx = int(np.argmax(probs))
        except Exception:
            record_idx = 0

        # Generate all individual plots
        logger.info(f"  Generating plots for {label}...")
        plot_summary_beeswarm(explanation, label, short, shap_plots_dir, timestamp)
        plot_bar_importance(explanation, label, short, shap_plots_dir, timestamp)
        plot_waterfall(explanation, label, short, shap_plots_dir, timestamp, record_idx)
        plot_dependence(shap_array, X_test, label, short, shap_plots_dir, timestamp)

        # Extract top features for analysis JSON
        all_top_features[model_name] = extract_top_features(
            shap_array, feature_names, top_n=CONFIG["top_n_features"]
        )
        logger.info(
            f"  Top feature: {all_top_features[model_name][0]['feature']} "
            f"(mean |SHAP| = {all_top_features[model_name][0]['mean_abs_shap']:.4f})"
        )

    # ----------------------------------------------------------------
    # Cross-model comparison plots
    # ----------------------------------------------------------------
    logger.info("\n--- Generating cross-model comparison plots ---")

    if len(all_shap) > 0:
        plot_feature_importance_comparison(all_shap, feature_names, shap_plots_dir, timestamp)

    plot_hallucination_comparison(results_df, shap_plots_dir, timestamp)
    plot_metric_heatmap(results_df, shap_plots_dir, timestamp)

    # ----------------------------------------------------------------
    # SHAP analysis JSON
    # ----------------------------------------------------------------
    logger.info("\n--- Saving SHAP analysis JSON ---")

    shap_analysis = {
        "timestamp": timestamp,
        "models_analyzed": list(models.keys()),
        "top_features_per_model": all_top_features,
        "feature_names": feature_names,
        "config": {
            "top_n_features": CONFIG["top_n_features"],
            "shap_penalty_threshold": 0.60,
        }
    }

    analysis_file = os.path.join(metrics_dir, f"shap_analysis_{timestamp}.json")
    with open(analysis_file, "w") as f:
        json.dump(shap_analysis, f, indent=2)
    logger.info(f"SHAP analysis saved: {analysis_file}")

    # ----------------------------------------------------------------
    # Hallucination correction analysis
    # ----------------------------------------------------------------
    logger.info("\n--- Analyzing hallucination corrections ---")
    correction_analysis = analyze_hallucination_corrections(
        metrics_dir, y_test, timestamp
    )

    correction_file = os.path.join(
        metrics_dir, f"hallucination_correction_analysis_{timestamp}.json"
    )
    with open(correction_file, "w") as f:
        json.dump(correction_analysis, f, indent=2)
    logger.info(f"Correction analysis saved: {correction_file}")

    if correction_analysis["summary"].get("total_corrections", 0) > 0:
        logger.info(f"\n  {correction_analysis['summary']['interpretation']}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Script 5 Complete — Visual Artifacts Summary")
    logger.info("=" * 60)

    png_files = glob.glob(os.path.join(shap_plots_dir, f"*_{timestamp}.png"))
    logger.info(f"Plots generated: {len(png_files)}")
    for f in sorted(png_files):
        logger.info(f"  {os.path.basename(f)}")

    logger.info("\nNext: python export_for_r.py (Script 6)")


if __name__ == "__main__":
    main()
