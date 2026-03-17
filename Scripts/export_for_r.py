"""
RA5olver - R Export & Statistical Analysis Preparation
========================================================
Script 6 of 6: Preparing All Data for R Statistical Analysis

Purpose:
    Transforms all experimental outputs from Scripts 1-5 into clean,
    well-structured CSVs optimized for R statistical analysis.
    Also generates the companion R analysis scripts that perform
    the actual hypothesis testing.

Research Context:
    Dissertation: "Design Science Approach to Explainable AI for Reducing
    Hallucinations in Vulnerability Management"
    Author: Ian Rivera | Colorado Technical University | 2025

R Analysis Scripts Generated:
    1. t_tests.R          — Independent samples t-tests (H0/H1)
    2. regression.R       — Multiple regression analysis
    3. visualizations.R   — Publication-ready R plots (ggplot2)
    4. assumptions.R      — Statistical assumption checks

CSV Exports Produced:
    1. ra5olver_main_results.csv      — Primary: all 6 conditions, all metrics
    2. ra5olver_paired_comparison.csv — Baseline vs XAI pairs per model
    3. ra5olver_hallucination_long.csv— Long format for mixed ANOVA / regression
    4. ra5olver_shap_features.csv     — Top SHAP features per model
    5. ra5olver_corrections.csv       — SHAP filter correction breakdown
    6. ra5olver_codebook.csv          — Variable definitions for dissertation

Statistical Tests Prepared:
    H0/H1 Test (Primary):
        Independent samples t-test:
        Baseline hallucination rates vs XAI hallucination rates
        Across all model types combined and per model type

    Regression (Secondary):
        DV  : hallucination_rate
        IVs : shap_enabled, model_type (dummy coded), interaction terms
        Tests whether SHAP is a significant predictor of hallucination
        reduction after controlling for model type

    Assumption Checks:
        - Shapiro-Wilk normality test
        - Levene's test for homogeneity of variance
        - Effect size (Cohen's d)
        - Power analysis

Usage:
    python export_for_r.py
        --results   Outputs/metrics/experiment_results_YYYYMMDD.csv
        --shap      Outputs/metrics/shap_analysis_YYYYMMDD.json
        --corrections Outputs/metrics/hallucination_correction_analysis_YYYYMMDD.json

Output:
    R_Analysis/data/ra5olver_*.csv       (R-ready data)
    R_Analysis/t_tests.R                 (hypothesis testing)
    R_Analysis/regression.R              (regression analysis)
    R_Analysis/visualizations.R          (ggplot2 figures)
    R_Analysis/assumptions.R             (assumption checks)
    R_Analysis/run_all.R                 (master runner)

Dependencies:
    pip install pandas numpy scipy
"""

import os
import json
import logging
import argparse
import textwrap
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    "r_analysis_dir": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "R_Analysis"
    ),
    "r_data_dir": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "R_Analysis", "data"
    ),
    "metrics_dir": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "Outputs", "metrics"
    ),
}

# ==============================================================================
# LOGGING
# ==============================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(output_dir: str, ts: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"export_log_{ts}.txt")
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
# DATA PREPARATION
# ==============================================================================

def prepare_main_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare primary results table.
    One row per experimental condition — 6 rows total.
    This is the master table all R analyses reference.
    """
    df = results_df.copy()

    # Ensure correct types
    df["shap_enabled"]   = df["shap_enabled"].astype(int)
    df["condition_id"]   = df["condition_id"].astype(int)

    # Add dummy-coded model type columns for regression
    df["model_lr"]  = (df["model_type"] == "LogisticRegression").astype(int)
    df["model_rf"]  = (df["model_type"] == "RandomForest").astype(int)
    df["model_svm"] = (df["model_type"] == "SVM").astype(int)

    # Add interaction terms (shap x model type)
    df["shap_x_lr"]  = df["shap_enabled"] * df["model_lr"]
    df["shap_x_rf"]  = df["shap_enabled"] * df["model_rf"]
    df["shap_x_svm"] = df["shap_enabled"] * df["model_svm"]

    # Readable condition label
    df["condition_label"] = df.apply(
        lambda r: f"{r['model_type']} ({'XAI' if r['shap_enabled'] else 'Baseline'})",
        axis=1
    )

    # Reorder columns for readability in R
    col_order = [
        "condition_id", "condition_label", "model_type",
        "shap_enabled", "model_lr", "model_rf", "model_svm",
        "shap_x_lr", "shap_x_rf", "shap_x_svm",
        "total_predictions", "true_positives", "true_negatives",
        "false_positives", "false_negatives", "total_hallucinations",
        "hallucination_rate", "false_positive_rate", "false_negative_rate",
        "precision", "recall", "f1_score", "accuracy", "roc_auc",
        "hallucination_reduction_vs_baseline", "hallucination_reduction_pct",
        "timestamp"
    ]
    existing = [c for c in col_order if c in df.columns]
    df = df[existing]

    logger.info(f"Main results table: {df.shape}")
    return df


def prepare_paired_comparison(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare paired baseline vs XAI comparison table.
    One row per model type — 3 rows total.
    Optimized for paired t-test input in R.

    Columns:
        model_type
        baseline_hallucination_rate
        xai_hallucination_rate
        difference (baseline - xai)
        reduction_pct
        baseline_fpr / xai_fpr
        baseline_f1 / xai_f1
        baseline_auc / xai_auc
    """
    pairs = []
    model_types = ["LogisticRegression", "RandomForest", "SVM"]

    for model in model_types:
        baseline = results_df[
            (results_df["model_type"] == model) &
            (results_df["shap_enabled"] == 0)
        ]
        xai = results_df[
            (results_df["model_type"] == model) &
            (results_df["shap_enabled"] == 1)
        ]

        if len(baseline) == 0 or len(xai) == 0:
            logger.warning(f"Missing data for {model} paired comparison.")
            continue

        b = baseline.iloc[0]
        x = xai.iloc[0]

        diff = float(b["hallucination_rate"]) - float(x["hallucination_rate"])
        pct  = (diff / float(b["hallucination_rate"]) * 100
                if float(b["hallucination_rate"]) > 0 else 0.0)

        pairs.append({
            "model_type": model,
            "baseline_condition_id": int(b["condition_id"]),
            "xai_condition_id": int(x["condition_id"]),

            # Primary outcome
            "baseline_hallucination_rate": round(float(b["hallucination_rate"]), 6),
            "xai_hallucination_rate": round(float(x["hallucination_rate"]), 6),
            "hallucination_difference": round(diff, 6),
            "hallucination_reduction_pct": round(pct, 2),

            # Secondary metrics
            "baseline_fpr": round(float(b["false_positive_rate"]), 6),
            "xai_fpr": round(float(x["false_positive_rate"]), 6),
            "fpr_difference": round(
                float(b["false_positive_rate"]) - float(x["false_positive_rate"]), 6
            ),
            "baseline_f1": round(float(b["f1_score"]), 6),
            "xai_f1": round(float(x["f1_score"]), 6),
            "f1_difference": round(
                float(x["f1_score"]) - float(b["f1_score"]), 6
            ),
            "baseline_auc": round(float(b["roc_auc"]) if pd.notna(b["roc_auc"]) else 0, 6),
            "xai_auc": round(float(x["roc_auc"]) if pd.notna(x["roc_auc"]) else 0, 6),
        })

    df = pd.DataFrame(pairs)
    logger.info(f"Paired comparison table: {df.shape}")
    return df


def prepare_long_format(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape results into long format for mixed ANOVA and regression.

    Long format: one row per metric per condition.
    Required for some R regression and visualization functions.
    """
    metric_cols = [
        "hallucination_rate", "false_positive_rate", "false_negative_rate",
        "precision", "recall", "f1_score", "accuracy", "roc_auc"
    ]

    id_cols = ["condition_id", "model_type", "shap_enabled", "condition_label"]
    existing_id = [c for c in id_cols if c in results_df.columns]

    long_rows = []
    for _, row in results_df.iterrows():
        for metric in metric_cols:
            if metric in results_df.columns and pd.notna(row.get(metric)):
                long_rows.append({
                    **{c: row[c] for c in existing_id},
                    "metric_name": metric,
                    "metric_value": float(row[metric]),
                    "is_primary_outcome": int(metric == "hallucination_rate"),
                })

    df = pd.DataFrame(long_rows)
    logger.info(f"Long format table: {df.shape}")
    return df


def prepare_shap_features(shap_analysis: dict) -> pd.DataFrame:
    """
    Flatten SHAP top features into a tidy DataFrame for R visualization.
    """
    rows = []
    for model_name, features in shap_analysis.get("top_features_per_model", {}).items():
        for feat in features:
            rows.append({
                "model_type": model_name,
                "rank": feat["rank"],
                "feature_name": feat["feature"],
                "mean_abs_shap": feat["mean_abs_shap"],
                "interpretation": feat["interpretation"],
            })
    df = pd.DataFrame(rows)
    logger.info(f"SHAP features table: {df.shape}")
    return df


def prepare_corrections(correction_analysis: dict) -> pd.DataFrame:
    """
    Flatten SHAP correction analysis into tidy DataFrame.
    """
    rows = []
    for cond in correction_analysis.get("conditions_analyzed", []):
        rows.append({
            "condition_id": cond["condition_id"],
            "total_corrections": cond["total_corrections"],
            "true_corrections": cond["true_corrections"],
            "overcorrections": cond["overcorrections"],
            "correction_accuracy": cond["correction_accuracy"],
        })

    summary = correction_analysis.get("summary", {})
    if summary:
        rows.append({
            "condition_id": "ALL",
            "total_corrections": summary.get("total_corrections", 0),
            "true_corrections": summary.get("true_corrections", 0),
            "overcorrections": summary.get("overcorrections", 0),
            "correction_accuracy": summary.get("overall_correction_accuracy", 0),
        })

    df = pd.DataFrame(rows)
    logger.info(f"Corrections table: {df.shape}")
    return df


def build_codebook() -> pd.DataFrame:
    """
    Generate variable codebook for dissertation appendix.
    Documents every variable used in R analysis.
    """
    codebook = [
        # Identifiers
        ("condition_id",        "integer",    "Experimental condition number (1-6)"),
        ("condition_label",     "character",  "Human-readable condition label"),
        ("model_type",          "character",  "ML model: LogisticRegression / RandomForest / SVM"),
        ("shap_enabled",        "integer",    "Treatment variable: 0=Baseline, 1=XAI (SHAP)"),

        # Dummy codes
        ("model_lr",            "integer",    "Dummy: 1 if LogisticRegression, else 0"),
        ("model_rf",            "integer",    "Dummy: 1 if RandomForest, else 0"),
        ("model_svm",           "integer",    "Dummy: 1 if SVM, else 0"),
        ("shap_x_lr",           "integer",    "Interaction: shap_enabled x model_lr"),
        ("shap_x_rf",           "integer",    "Interaction: shap_enabled x model_rf"),
        ("shap_x_svm",          "integer",    "Interaction: shap_enabled x model_svm"),

        # Confusion matrix counts
        ("true_positives",      "integer",    "Correct High priority predictions"),
        ("true_negatives",      "integer",    "Correct Low priority predictions"),
        ("false_positives",     "integer",    "Low predicted as High (type I error)"),
        ("false_negatives",     "integer",    "High predicted as Low (type II error)"),
        ("total_hallucinations","integer",    "FP + FN — total incorrect predictions"),

        # Primary outcome (H0/H1)
        ("hallucination_rate",  "numeric",
         "PRIMARY DV: (FP+FN)/Total. Operationalized hallucination rate. "
         "H1: XAI reduces this vs baseline."),

        # Secondary metrics
        ("false_positive_rate", "numeric",    "FP / (FP + TN) — false alarm rate"),
        ("false_negative_rate", "numeric",    "FN / (FN + TP) — miss rate"),
        ("precision",           "numeric",    "TP / (TP + FP)"),
        ("recall",              "numeric",    "TP / (TP + FN) — sensitivity"),
        ("f1_score",            "numeric",    "Harmonic mean of precision and recall"),
        ("accuracy",            "numeric",    "Correct predictions / Total predictions"),
        ("roc_auc",             "numeric",    "Area Under ROC Curve (0.5-1.0)"),

        # Reduction metrics
        ("hallucination_reduction_vs_baseline", "numeric",
         "Baseline rate minus XAI rate (positive = improvement)"),
        ("hallucination_reduction_pct", "numeric",
         "Percentage reduction in hallucination rate vs paired baseline"),

        # SHAP
        ("mean_abs_shap",       "numeric",    "Mean absolute SHAP value — feature importance"),
        ("correction_accuracy", "numeric",
         "Proportion of SHAP filter reclassifications that were correct"),
    ]

    df = pd.DataFrame(codebook, columns=["variable", "type", "description"])
    logger.info(f"Codebook: {len(df)} variables documented")
    return df


# ==============================================================================
# DESCRIPTIVE STATISTICS (Python preview of R results)
# ==============================================================================

def compute_descriptive_stats(results_df: pd.DataFrame) -> str:
    """
    Compute descriptive statistics as a preview.
    Mirrors what R will produce — useful for sanity checking.
    """
    baseline = results_df[results_df["shap_enabled"] == 0]["hallucination_rate"]
    xai      = results_df[results_df["shap_enabled"] == 1]["hallucination_rate"]

    # Independent samples t-test preview
    if len(baseline) >= 2 and len(xai) >= 2:
        t_stat, p_val = stats.ttest_ind(baseline, xai, alternative="greater")
        # Cohen's d
        pooled_std = np.sqrt(
            ((len(baseline)-1)*baseline.std()**2 + (len(xai)-1)*xai.std()**2) /
            (len(baseline) + len(xai) - 2)
        )
        cohens_d = (baseline.mean() - xai.mean()) / pooled_std if pooled_std > 0 else 0
    else:
        t_stat, p_val, cohens_d = None, None, None

    stats_text = f"""
DESCRIPTIVE STATISTICS PREVIEW (Python)
-----------------------------------------
Baseline Hallucination Rates:
  N        : {len(baseline)}
  Mean     : {baseline.mean():.4f}
  Std Dev  : {baseline.std():.4f}
  Min      : {baseline.min():.4f}
  Max      : {baseline.max():.4f}

XAI Hallucination Rates:
  N        : {len(xai)}
  Mean     : {xai.mean():.4f}
  Std Dev  : {xai.std():.4f}
  Min      : {xai.min():.4f}
  Max      : {xai.max():.4f}

T-Test Preview (one-tailed: baseline > xai):
  t-statistic : {f"{t_stat:.4f}" if t_stat is not None else "N/A"}
  p-value     : {f"{p_val:.4f}" if p_val is not None else "N/A"}
  Cohen's d   : {f"{cohens_d:.4f}" if cohens_d is not None else "N/A"}

  {"NOTE: p < 0.05 — preliminary evidence supports H1" if p_val and p_val < 0.05
   else "NOTE: p >= 0.05 — preliminary evidence does not support H1"
   if p_val is not None else ""}
  (Confirm with full R analysis — see R_Analysis/t_tests.R)
"""
    return stats_text


# ==============================================================================
# R SCRIPT GENERATION
# ==============================================================================

def generate_t_tests_r(data_dir: str) -> str:
    """Generate t_tests.R — primary H0/H1 hypothesis testing script."""
    return textwrap.dedent(f'''
    # ============================================================
    # RA5olver - t_tests.R
    # Hypothesis Testing: H0/H1 Independent Samples T-Tests
    # ============================================================
    # Dissertation: Design Science Approach to Explainable AI
    # Author: Ian Rivera | Colorado Technical University | 2025
    #
    # H0: SHAP-based XAI cannot significantly reduce hallucinated
    #     outputs vs baseline (hallucination_rate baseline = xai)
    # H1: SHAP-based XAI can significantly reduce hallucinated
    #     outputs vs baseline (hallucination_rate baseline > xai)
    # ============================================================

    library(dplyr)
    library(effsize)   # For Cohen's d: install.packages("effsize")
    library(pwr)       # For power analysis: install.packages("pwr")

    # Load data
    data_dir <- "{data_dir.replace(os.sep, "/")}"
    df_main  <- read.csv(file.path(data_dir, "ra5olver_main_results.csv"))
    df_pairs <- read.csv(file.path(data_dir, "ra5olver_paired_comparison.csv"))

    cat("\\n============================================================\\n")
    cat("RA5olver Hypothesis Testing\\n")
    cat("============================================================\\n")

    # ------------------------------------------------------------
    # TEST 1: Overall Independent Samples T-Test
    # Compares all baseline conditions vs all XAI conditions
    # ------------------------------------------------------------
    cat("\\nTEST 1: Overall Baseline vs XAI (Independent Samples t-test)\\n")
    cat("------------------------------------------------------------\\n")

    baseline_rates <- df_main %>%
      filter(shap_enabled == 0) %>%
      pull(hallucination_rate)

    xai_rates <- df_main %>%
      filter(shap_enabled == 1) %>%
      pull(hallucination_rate)

    cat("Baseline hallucination rates:", baseline_rates, "\\n")
    cat("XAI hallucination rates:     ", xai_rates, "\\n")
    cat("\\nDescriptive Statistics:\\n")
    cat("  Baseline Mean:", round(mean(baseline_rates), 4), "\\n")
    cat("  XAI Mean:     ", round(mean(xai_rates), 4), "\\n")
    cat("  Difference:   ", round(mean(baseline_rates) - mean(xai_rates), 4), "\\n")

    # One-tailed t-test: baseline > xai (directional H1)
    t_result_overall <- t.test(
      baseline_rates, xai_rates,
      alternative = "greater",
      var.equal   = FALSE   # Welch t-test (conservative)
    )
    print(t_result_overall)

    # Effect size (Cohen's d)
    d_overall <- cohen.d(baseline_rates, xai_rates)
    cat("\\nEffect Size (Cohen\\'s d):", round(d_overall$estimate, 4), "\\n")
    cat("Magnitude:", d_overall$magnitude, "\\n")

    # Decision
    alpha <- 0.05
    if (t_result_overall$p.value < alpha) {{
      cat("\\nDECISION: REJECT H0 — Evidence supports H1\\n")
      cat("  XAI significantly reduces hallucination rate (p <", alpha, ")\\n")
    }} else {{
      cat("\\nDECISION: FAIL TO REJECT H0\\n")
      cat("  Insufficient evidence to support H1 (p =",
          round(t_result_overall$p.value, 4), ")\\n")
    }}

    # ------------------------------------------------------------
    # TEST 2: Per-Model T-Tests
    # ------------------------------------------------------------
    cat("\\n\\nTEST 2: Per-Model Paired Comparisons\\n")
    cat("------------------------------------------------------------\\n")

    models <- unique(df_main$model_type)
    for (model in models) {{
      cat("\\nModel:", model, "\\n")
      b_rate <- df_main %>% filter(model_type == model, shap_enabled == 0) %>%
                pull(hallucination_rate)
      x_rate <- df_main %>% filter(model_type == model, shap_enabled == 1) %>%
                pull(hallucination_rate)
      cat("  Baseline:", round(b_rate, 4),
          "| XAI:", round(x_rate, 4),
          "| Reduction:", round(b_rate - x_rate, 4),
          "(", round((b_rate - x_rate)/b_rate * 100, 1), "% )\\n")
    }}

    # Paired t-test across model pairs
    cat("\\nPaired T-Test (baseline vs xai across 3 model pairs):\\n")
    t_paired <- t.test(
      df_pairs$baseline_hallucination_rate,
      df_pairs$xai_hallucination_rate,
      paired      = TRUE,
      alternative = "greater"
    )
    print(t_paired)

    # Effect size for paired test
    d_paired <- cohen.d(
      df_pairs$baseline_hallucination_rate,
      df_pairs$xai_hallucination_rate,
      paired = TRUE
    )
    cat("\\nPaired Effect Size (Cohen\\'s d):", round(d_paired$estimate, 4), "\\n")

    # ------------------------------------------------------------
    # TEST 3: Secondary Metrics T-Tests
    # ------------------------------------------------------------
    cat("\\n\\nTEST 3: Secondary Metrics — Baseline vs XAI\\n")
    cat("------------------------------------------------------------\\n")

    secondary_metrics <- c("false_positive_rate", "f1_score", "accuracy", "roc_auc")
    for (metric in secondary_metrics) {{
      b_vals <- df_main %>% filter(shap_enabled == 0) %>% pull(!!sym(metric))
      x_vals <- df_main %>% filter(shap_enabled == 1) %>% pull(!!sym(metric))
      b_vals <- b_vals[!is.na(b_vals)]
      x_vals <- x_vals[!is.na(x_vals)]
      if (length(b_vals) >= 2 & length(x_vals) >= 2) {{
        direction <- if (metric %in% c("false_positive_rate")) "greater" else "less"
        t_sec <- t.test(b_vals, x_vals, alternative = direction, var.equal = FALSE)
        cat("\\n", metric, ":\\n")
        cat("  Baseline Mean:", round(mean(b_vals), 4),
            "| XAI Mean:", round(mean(x_vals), 4),
            "| p-value:", round(t_sec$p.value, 4), "\\n")
      }}
    }}

    # ------------------------------------------------------------
    # Power Analysis
    # ------------------------------------------------------------
    cat("\\n\\nPOWER ANALYSIS\\n")
    cat("------------------------------------------------------------\\n")
    # Post-hoc power given observed effect size
    if (!is.null(d_overall$estimate)) {{
      pwr_result <- pwr.t.test(
        n           = length(baseline_rates),
        d           = abs(d_overall$estimate),
        sig.level   = 0.05,
        type        = "two.sample",
        alternative = "greater"
      )
      cat("Post-hoc Power:", round(pwr_result$power, 4), "\\n")
      cat("  (Power >= 0.80 is conventionally acceptable)\\n")
    }}

    cat("\\n============================================================\\n")
    cat("T-Test Analysis Complete\\n")
    cat("============================================================\\n")
    ''')


def generate_regression_r(data_dir: str) -> str:
    """Generate regression.R — multiple regression analysis script."""
    return textwrap.dedent(f'''
    # ============================================================
    # RA5olver - regression.R
    # Multiple Regression: Predictors of Hallucination Rate
    # ============================================================
    # DV : hallucination_rate
    # IVs: shap_enabled, model type (dummy coded), interactions
    #
    # Research Question:
    #   Is SHAP integration a significant predictor of reduced
    #   hallucination rate after controlling for model type?
    # ============================================================

    library(dplyr)
    library(car)      # For VIF: install.packages("car")
    library(lmtest)   # For Breusch-Pagan: install.packages("lmtest")

    data_dir <- "{data_dir.replace(os.sep, "/")}"
    df <- read.csv(file.path(data_dir, "ra5olver_main_results.csv"))

    cat("\\n============================================================\\n")
    cat("RA5olver Regression Analysis\\n")
    cat("============================================================\\n")

    # ------------------------------------------------------------
    # MODEL 1: Simple regression — SHAP only
    # ------------------------------------------------------------
    cat("\\nMODEL 1: Simple Regression (SHAP only)\\n")
    cat("------------------------------------------------------------\\n")
    model1 <- lm(hallucination_rate ~ shap_enabled, data = df)
    print(summary(model1))

    # ------------------------------------------------------------
    # MODEL 2: Multiple regression — SHAP + model type
    # ------------------------------------------------------------
    cat("\\nMODEL 2: Multiple Regression (SHAP + model type)\\n")
    cat("------------------------------------------------------------\\n")
    # Note: model_svm omitted as reference category
    model2 <- lm(hallucination_rate ~ shap_enabled + model_lr + model_rf,
                 data = df)
    print(summary(model2))

    # ------------------------------------------------------------
    # MODEL 3: Full model with interaction terms
    # ------------------------------------------------------------
    cat("\\nMODEL 3: Full Model with Interactions\\n")
    cat("------------------------------------------------------------\\n")
    model3 <- lm(
      hallucination_rate ~ shap_enabled + model_lr + model_rf +
        shap_x_lr + shap_x_rf,
      data = df
    )
    print(summary(model3))

    # ------------------------------------------------------------
    # MODEL COMPARISON
    # ------------------------------------------------------------
    cat("\\nMODEL COMPARISON (ANOVA)\\n")
    cat("------------------------------------------------------------\\n")
    print(anova(model1, model2, model3))

    # ------------------------------------------------------------
    # REGRESSION DIAGNOSTICS
    # ------------------------------------------------------------
    cat("\\nREGRESSION DIAGNOSTICS — Model 2\\n")
    cat("------------------------------------------------------------\\n")

    # VIF (multicollinearity)
    cat("\\nVariance Inflation Factors (VIF):\\n")
    tryCatch(print(vif(model2)), error = function(e) cat("VIF not available\\n"))

    # Breusch-Pagan (homoscedasticity)
    cat("\\nBreusch-Pagan Test (homoscedasticity):\\n")
    tryCatch(print(bptest(model2)), error = function(e) cat("BP test not available\\n"))

    # Residual normality
    cat("\\nShapiro-Wilk Test on Residuals:\\n")
    sw <- shapiro.test(residuals(model2))
    print(sw)
    if (sw$p.value > 0.05) {{
      cat("  Residuals appear normally distributed (p > 0.05)\\n")
    }} else {{
      cat("  Residuals may not be normally distributed (p <= 0.05)\\n")
      cat("  Consider bootstrap confidence intervals\\n")
    }}

    # ------------------------------------------------------------
    # KEY FINDING: SHAP coefficient interpretation
    # ------------------------------------------------------------
    cat("\\nKEY FINDING: SHAP Coefficient (Model 2)\\n")
    cat("------------------------------------------------------------\\n")
    shap_coef <- coef(model2)["shap_enabled"]
    shap_p    <- summary(model2)$coefficients["shap_enabled", "Pr(>|t|)"]
    cat("SHAP coefficient:", round(shap_coef, 6), "\\n")
    cat("p-value:         ", round(shap_p, 4), "\\n")
    cat("Interpretation:   A one-unit increase in shap_enabled (i.e., activating\\n")
    cat("  SHAP XAI integration) is associated with a",
        round(shap_coef, 4), "change\\n")
    cat("  in hallucination rate, controlling for model type.\\n")
    if (shap_p < 0.05) {{
      cat("  This effect is statistically significant (p < 0.05).\\n")
    }} else {{
      cat("  This effect is not statistically significant at alpha=0.05.\\n")
    }}

    cat("\\n============================================================\\n")
    cat("Regression Analysis Complete\\n")
    cat("============================================================\\n")
    ''')


def generate_assumptions_r(data_dir: str) -> str:
    """Generate assumptions.R — statistical assumption verification."""
    return textwrap.dedent(f'''
    # ============================================================
    # RA5olver - assumptions.R
    # Statistical Assumption Checks Before Hypothesis Testing
    # ============================================================
    # Run BEFORE t_tests.R and regression.R
    # Document results in dissertation methodology section
    # ============================================================

    library(dplyr)
    library(car)

    data_dir <- "{data_dir.replace(os.sep, "/")}"
    df <- read.csv(file.path(data_dir, "ra5olver_main_results.csv"))

    cat("\\n============================================================\\n")
    cat("RA5olver Statistical Assumption Checks\\n")
    cat("============================================================\\n")

    baseline <- df %>% filter(shap_enabled == 0) %>% pull(hallucination_rate)
    xai      <- df %>% filter(shap_enabled == 1) %>% pull(hallucination_rate)

    # ------------------------------------------------------------
    # ASSUMPTION 1: Normality (Shapiro-Wilk)
    # ------------------------------------------------------------
    cat("\\nASSUMPTION 1: Normality (Shapiro-Wilk Test)\\n")
    cat("------------------------------------------------------------\\n")
    cat("Null: Data is normally distributed\\n\\n")

    cat("Baseline group:\\n")
    if (length(baseline) >= 3) {{
      sw_b <- shapiro.test(baseline)
      print(sw_b)
    }} else {{
      cat("  Sample too small for Shapiro-Wilk (n =", length(baseline), ")\\n")
      cat("  NOTE: With n=3 per group, normality cannot be formally tested.\\n")
      cat("  Document as study limitation. Consider non-parametric alternative.\\n")
    }}

    cat("\\nXAI group:\\n")
    if (length(xai) >= 3) {{
      sw_x <- shapiro.test(xai)
      print(sw_x)
    }} else {{
      cat("  Sample too small for Shapiro-Wilk (n =", length(xai), ")\\n")
    }}

    # ------------------------------------------------------------
    # ASSUMPTION 2: Homogeneity of Variance (Levene's Test)
    # ------------------------------------------------------------
    cat("\\nASSUMPTION 2: Homogeneity of Variance (Levene\\'s Test)\\n")
    cat("------------------------------------------------------------\\n")
    cat("Null: Variances are equal across groups\\n\\n")

    df$shap_factor <- as.factor(df$shap_enabled)
    tryCatch(
      {{
        levene <- leveneTest(hallucination_rate ~ shap_factor, data = df)
        print(levene)
        if (levene[1, "Pr(>F)"] > 0.05) {{
          cat("Variances appear equal (p > 0.05) — t-test assumption met\\n")
        }} else {{
          cat("Variances may differ (p <= 0.05) — use Welch t-test (var.equal=FALSE)\\n")
          cat("NOTE: t_tests.R already uses Welch t-test by default\\n")
        }}
      }},
      error = function(e) cat("Levene\\'s test requires car package\\n")
    )

    # ------------------------------------------------------------
    # ASSUMPTION 3: Independence
    # ------------------------------------------------------------
    cat("\\nASSUMPTION 3: Independence of Observations\\n")
    cat("------------------------------------------------------------\\n")
    cat("Each experimental condition uses the same held-out test set.\\n")
    cat("Conditions are independent (different model instances).\\n")
    cat("The paired t-test in t_tests.R accounts for model-level pairing.\\n")
    cat("Independence assumption: MET by design\\n")

    # ------------------------------------------------------------
    # SMALL SAMPLE NOTE
    # ------------------------------------------------------------
    cat("\\nSMALL SAMPLE ADVISORY\\n")
    cat("------------------------------------------------------------\\n")
    cat("n = 3 per group (one observation per model type per condition)\\n")
    cat("This is a known limitation of the 6-condition design.\\n")
    cat("\\nRecommendations for dissertation:\\n")
    cat("  1. Report Welch t-test results (robust to unequal variance)\\n")
    cat("  2. Report effect sizes (Cohen\\'s d) alongside p-values\\n")
    cat("  3. Consider non-parametric Wilcoxon signed-rank test\\n")
    cat("     as sensitivity analysis\\n")
    cat("  4. Document as limitation: small n limits statistical power\\n")
    cat("  5. Future work: larger condition sets\\n")

    # Wilcoxon as non-parametric alternative
    cat("\\nNon-Parametric Alternative: Wilcoxon Signed-Rank Test\\n")
    cat("------------------------------------------------------------\\n")
    tryCatch(
      {{
        wilcox <- wilcox.test(baseline, xai, alternative = "greater", paired = FALSE)
        print(wilcox)
      }},
      error = function(e) cat("Wilcoxon test failed:", conditionMessage(e), "\\n")
    )

    cat("\\n============================================================\\n")
    cat("Assumption Checks Complete\\n")
    cat("Run t_tests.R and regression.R for hypothesis testing\\n")
    cat("============================================================\\n")
    ''')


def generate_visualizations_r(data_dir: str, plots_dir: str) -> str:
    """Generate visualizations.R — ggplot2 publication figures."""
    return textwrap.dedent(f'''
    # ============================================================
    # RA5olver - visualizations.R
    # Publication-Quality Figures (ggplot2)
    # ============================================================

    library(ggplot2)
    library(dplyr)
    library(tidyr)
    library(scales)

    data_dir  <- "{data_dir.replace(os.sep, "/")}"
    plots_dir <- "{plots_dir.replace(os.sep, "/")}"
    dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

    df       <- read.csv(file.path(data_dir, "ra5olver_main_results.csv"))
    df_pairs <- read.csv(file.path(data_dir, "ra5olver_paired_comparison.csv"))
    df_shap  <- read.csv(file.path(data_dir, "ra5olver_shap_features.csv"))
    df_long  <- read.csv(file.path(data_dir, "ra5olver_hallucination_long.csv"))

    # Labels
    df$condition_label <- factor(df$condition_label)
    df$shap_label <- ifelse(df$shap_enabled == 1, "XAI (SHAP)", "Baseline")
    df$model_label <- recode(df$model_type,
      "LogisticRegression" = "Logistic\\nRegression",
      "RandomForest"       = "Random\\nForest",
      "SVM"                = "SVM"
    )

    # Color palette (colorblind-friendly)
    palette <- c("Baseline" = "#4878CF", "XAI (SHAP)" = "#6ACC65")

    # ----------------------------------------------------------
    # FIGURE 1: Hallucination Rate — Primary Results
    # ----------------------------------------------------------
    fig1 <- ggplot(df, aes(x = model_label, y = hallucination_rate,
                           fill = shap_label)) +
      geom_bar(stat = "identity", position = position_dodge(width = 0.7),
               width = 0.6, alpha = 0.88) +
      geom_text(aes(label = sprintf("%.3f", hallucination_rate)),
                position = position_dodge(width = 0.7),
                vjust = -0.5, size = 3.5, color = "gray30") +
      scale_fill_manual(values = palette, name = "Condition") +
      scale_y_continuous(labels = scales::percent_format(accuracy = 0.1),
                         expand = expansion(mult = c(0, 0.15))) +
      labs(
        title    = "Hallucination Rate: Baseline vs XAI by Model Type",
        subtitle = "Primary Outcome Variable — H0/H1 Test",
        x        = "Model Type",
        y        = "Hallucination Rate\\n(Incorrect Predictions / Total)",
        caption  = "Lower is better. RA5olver Dissertation Study."
      ) +
      theme_minimal(base_size = 13) +
      theme(
        plot.title    = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 11, color = "gray40"),
        legend.position = "top",
        panel.grid.major.x = element_blank()
      )

    ggsave(file.path(plots_dir, "fig1_hallucination_rate.png"),
           fig1, width = 9, height = 6, dpi = 300)
    cat("Saved: fig1_hallucination_rate.png\\n")

    # ----------------------------------------------------------
    # FIGURE 2: All Metrics Comparison (Faceted)
    # ----------------------------------------------------------
    df_metrics_long <- df %>%
      select(model_label, shap_label,
             hallucination_rate, false_positive_rate,
             precision, recall, f1_score, accuracy) %>%
      pivot_longer(cols = -c(model_label, shap_label),
                   names_to = "metric", values_to = "value") %>%
      mutate(metric = recode(metric,
        "hallucination_rate"  = "Hallucination Rate",
        "false_positive_rate" = "False Positive Rate",
        "precision"           = "Precision",
        "recall"              = "Recall",
        "f1_score"            = "F1 Score",
        "accuracy"            = "Accuracy"
      ))

    fig2 <- ggplot(df_metrics_long,
                   aes(x = model_label, y = value, fill = shap_label)) +
      geom_bar(stat = "identity", position = position_dodge(width = 0.7),
               width = 0.6, alpha = 0.88) +
      facet_wrap(~ metric, scales = "free_y", ncol = 3) +
      scale_fill_manual(values = palette, name = "Condition") +
      labs(
        title   = "All Metrics: Baseline vs XAI Across Model Types",
        x       = "Model Type",
        y       = "Metric Value",
        caption = "RA5olver Dissertation Study"
      ) +
      theme_minimal(base_size = 11) +
      theme(
        plot.title      = element_text(face = "bold", size = 13),
        legend.position = "top",
        strip.text      = element_text(face = "bold"),
        panel.grid.major.x = element_blank()
      )

    ggsave(file.path(plots_dir, "fig2_all_metrics.png"),
           fig2, width = 13, height = 9, dpi = 300)
    cat("Saved: fig2_all_metrics.png\\n")

    # ----------------------------------------------------------
    # FIGURE 3: SHAP Feature Importance (Top 10 per Model)
    # ----------------------------------------------------------
    if (nrow(df_shap) > 0) {{
      df_shap_top <- df_shap %>%
        group_by(model_type) %>%
        slice_max(mean_abs_shap, n = 10) %>%
        ungroup() %>%
        mutate(
          feature_short = substr(feature_name, 1, 25),
          model_label   = recode(model_type,
            "LogisticRegression" = "Logistic Regression",
            "RandomForest"       = "Random Forest",
            "SVM"                = "SVM"
          )
        )

      fig3 <- ggplot(df_shap_top,
                     aes(x = reorder(feature_short, mean_abs_shap),
                         y = mean_abs_shap, fill = model_label)) +
        geom_bar(stat = "identity", alpha = 0.85) +
        coord_flip() +
        facet_wrap(~ model_label, scales = "free_x") +
        scale_fill_brewer(palette = "Set2") +
        labs(
          title   = "Top 10 SHAP Features by Model",
          subtitle = "Mean |SHAP Value| — Higher = More Influential",
          x       = "Feature",
          y       = "Mean |SHAP Value|",
          caption = "RA5olver Dissertation Study"
        ) +
        theme_minimal(base_size = 11) +
        theme(
          plot.title      = element_text(face = "bold"),
          legend.position = "none",
          strip.text      = element_text(face = "bold")
        )

      ggsave(file.path(plots_dir, "fig3_shap_features.png"),
             fig3, width = 14, height = 8, dpi = 300)
      cat("Saved: fig3_shap_features.png\\n")
    }}

    # ----------------------------------------------------------
    # FIGURE 4: Hallucination Reduction Summary
    # ----------------------------------------------------------
    df_pairs$model_label <- recode(df_pairs$model_type,
      "LogisticRegression" = "Logistic\\nRegression",
      "RandomForest"       = "Random\\nForest",
      "SVM"                = "SVM"
    )

    fig4 <- ggplot(df_pairs,
                   aes(x = reorder(model_label, -hallucination_reduction_pct),
                       y = hallucination_reduction_pct,
                       fill = hallucination_reduction_pct > 0)) +
      geom_bar(stat = "identity", width = 0.5, alpha = 0.88) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
      geom_text(aes(label = paste0(round(hallucination_reduction_pct, 1), "%")),
                vjust = -0.5, size = 4, fontface = "bold") +
      scale_fill_manual(values = c("TRUE" = "#6ACC65", "FALSE" = "#D65F5F"),
                        guide = "none") +
      labs(
        title    = "Hallucination Reduction: XAI vs Baseline",
        subtitle = "Positive = XAI reduced hallucinations vs paired baseline",
        x        = "Model Type",
        y        = "Hallucination Rate Reduction (%)",
        caption  = "RA5olver Dissertation Study"
      ) +
      theme_minimal(base_size = 13) +
      theme(
        plot.title      = element_text(face = "bold", size = 14),
        plot.subtitle   = element_text(size = 11, color = "gray40"),
        panel.grid.major.x = element_blank()
      )

    ggsave(file.path(plots_dir, "fig4_hallucination_reduction.png"),
           fig4, width = 9, height = 6, dpi = 300)
    cat("Saved: fig4_hallucination_reduction.png\\n")

    cat("\\nAll R visualizations saved to:", plots_dir, "\\n")
    ''')


def generate_run_all_r(r_dir: str) -> str:
    """Generate run_all.R — master runner for full R analysis."""
    return textwrap.dedent(f'''
    # ============================================================
    # RA5olver - run_all.R
    # Master Runner — Execute Full R Analysis Pipeline
    # ============================================================
    # Run this script to reproduce all statistical results
    # ============================================================

    r_dir <- "{r_dir.replace(os.sep, "/")}"

    cat("\\n============================================================\\n")
    cat("RA5olver R Analysis Pipeline\\n")
    cat("============================================================\\n")

    # Step 1: Check assumptions first
    cat("\\n[1/4] Running assumption checks...\\n")
    source(file.path(r_dir, "assumptions.R"))

    # Step 2: Hypothesis testing
    cat("\\n[2/4] Running t-tests (H0/H1)...\\n")
    source(file.path(r_dir, "t_tests.R"))

    # Step 3: Regression analysis
    cat("\\n[3/4] Running regression analysis...\\n")
    source(file.path(r_dir, "regression.R"))

    # Step 4: Generate visualizations
    cat("\\n[4/4] Generating visualizations...\\n")
    source(file.path(r_dir, "visualizations.R"))

    cat("\\n============================================================\\n")
    cat("RA5olver R Analysis Complete\\n")
    cat("All outputs saved to R_Analysis/\\n")
    cat("============================================================\\n")
    ''')


# ==============================================================================
# MAIN
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="RA5olver Script 6: Export for R Statistical Analysis"
    )
    parser.add_argument("--results",
                        required=True,
                        help="experiment_results CSV from Script 4")
    parser.add_argument("--shap",
                        required=False,
                        default=None,
                        help="shap_analysis JSON from Script 5")
    parser.add_argument("--corrections",
                        required=False,
                        default=None,
                        help="hallucination_correction_analysis JSON from Script 5")
    return parser.parse_args()


def main():
    global logger, timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    r_dir      = CONFIG["r_analysis_dir"]
    r_data_dir = CONFIG["r_data_dir"]
    r_plots_dir = os.path.join(r_dir, "plots")

    os.makedirs(r_data_dir, exist_ok=True)
    os.makedirs(r_plots_dir, exist_ok=True)

    logger = setup_logging(r_data_dir, timestamp)

    logger.info("=" * 60)
    logger.info("RA5olver - Script 6: Export for R Analysis")
    logger.info("=" * 60)

    args = parse_args()

    # Load results
    logger.info(f"Loading experiment results: {args.results}")
    results_df = pd.read_csv(args.results)
    logger.info(f"Loaded {len(results_df)} experimental conditions")

    # Load optional SHAP analysis
    shap_analysis = {}
    if args.shap and os.path.exists(args.shap):
        with open(args.shap) as f:
            shap_analysis = json.load(f)
        logger.info(f"Loaded SHAP analysis: {args.shap}")

    # Load optional correction analysis
    correction_analysis = {}
    if args.corrections and os.path.exists(args.corrections):
        with open(args.corrections) as f:
            correction_analysis = json.load(f)
        logger.info(f"Loaded correction analysis: {args.corrections}")

    # ----------------------------------------------------------------
    # Prepare all CSV exports
    # ----------------------------------------------------------------
    logger.info("\nPreparing R-ready data exports...")

    exports = {
        "ra5olver_main_results.csv":      prepare_main_results(results_df),
        "ra5olver_paired_comparison.csv": prepare_paired_comparison(results_df),
        "ra5olver_hallucination_long.csv":prepare_long_format(results_df),
        "ra5olver_shap_features.csv":     prepare_shap_features(shap_analysis),
        "ra5olver_corrections.csv":       prepare_corrections(correction_analysis),
        "ra5olver_codebook.csv":          build_codebook(),
    }

    for filename, df in exports.items():
        fpath = os.path.join(r_data_dir, filename)
        df.to_csv(fpath, index=False)
        logger.info(f"  Saved: {filename} ({df.shape[0]} rows x {df.shape[1]} cols)")

    # ----------------------------------------------------------------
    # Descriptive statistics preview
    # ----------------------------------------------------------------
    stats_text = compute_descriptive_stats(results_df)
    print(stats_text)

    stats_file = os.path.join(r_data_dir, f"descriptive_stats_{timestamp}.txt")
    with open(stats_file, "w") as f:
        f.write(stats_text)

    # ----------------------------------------------------------------
    # Generate R scripts
    # ----------------------------------------------------------------
    logger.info("\nGenerating R analysis scripts...")

    r_scripts = {
        "assumptions.R":    generate_assumptions_r(r_data_dir),
        "t_tests.R":        generate_t_tests_r(r_data_dir),
        "regression.R":     generate_regression_r(r_data_dir),
        "visualizations.R": generate_visualizations_r(r_data_dir, r_plots_dir),
        "run_all.R":        generate_run_all_r(r_dir),
    }

    for filename, content in r_scripts.items():
        fpath = os.path.join(r_dir, filename)
        with open(fpath, "w") as f:
            f.write(content)
        logger.info(f"  Generated: {filename}")

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Script 6 Complete — Pipeline Summary")
    logger.info("=" * 60)
    logger.info(f"\nR data exports : {r_data_dir}")
    logger.info(f"R scripts      : {r_dir}")
    logger.info(f"\nTo run full R analysis:")
    logger.info(f"  Open R or RStudio")
    logger.info(f"  source('{os.path.join(r_dir, 'run_all.R')}')")
    logger.info(f"\nRequired R packages:")
    logger.info(f"  install.packages(c('dplyr','ggplot2','tidyr','scales',")
    logger.info(f"                     'effsize','pwr','car','lmtest'))")
    logger.info(f"\nPipeline complete. All 6 scripts executed.")
    logger.info(f"Dissertation artifact is reproducible from:")
    logger.info(f"  python collect_nvd_data.py")
    logger.info(f"  -> python build_ground_truth.py")
    logger.info(f"  -> python feature_engineering.py")
    logger.info(f"  -> python train_models.py")
    logger.info(f"  -> python run_experiments.py")
    logger.info(f"  -> python export_for_r.py")
    logger.info(f"  -> Rscript run_all.R")


if __name__ == "__main__":
    main()
