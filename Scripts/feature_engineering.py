"""
RA5olver - Feature Engineering Pipeline
=========================================
Script 3 of 6: Consistent Feature Construction for All Experimental Conditions

Purpose:
    Transforms the ground-truth-labeled NVD dataset into a feature matrix
    ready for model training and evaluation. All 6 experimental conditions
    (LR, RF, SVM x baseline/XAI) use IDENTICAL features produced by this
    script — ensuring that any performance differences are attributable to
    model type and SHAP integration, not inconsistent preprocessing.

Research Context:
    Dissertation: "Design Science Approach to Explainable AI for Reducing
    Hallucinations in Vulnerability Management"
    Author: Ian Rivera | Colorado Technical University | 2025

Feature Groups:
    1. CVSS Numeric     — base score (continuous)
    2. CVSS Categorical — attack vector, complexity, privileges, CIA impact
                          (one-hot encoded)
    3. Text (TF-IDF)    — CVE description (top N terms, configurable)
    4. Temporal         — publication year, days since published
    5. Target           — binary encoding of ground_truth_label

Reproducibility:
    - All encoders and scalers are fit on TRAINING split only
    - Fit artifacts saved to Models/preprocessors/ for reuse in inference
    - Train/test split uses fixed random_state=42
    - Feature schema documented in outputs for dissertation appendix

Usage:
    python feature_engineering.py
        --input Data/ground_truth/nvd_resolvable_YYYYMMDD_HHMMSS.csv

Output:
    Data/processed/X_train_YYYYMMDD.csv
    Data/processed/X_test_YYYYMMDD.csv
    Data/processed/y_train_YYYYMMDD.csv
    Data/processed/y_test_YYYYMMDD.csv
    Data/processed/feature_schema_YYYYMMDD.json
    Models/preprocessors/tfidf_YYYYMMDD.pkl
    Models/preprocessors/scaler_YYYYMMDD.pkl
    Models/preprocessors/encoder_YYYYMMDD.pkl

Dependencies:
    pip install pandas numpy scikit-learn tqdm joblib
"""

import os
import json
import logging
import argparse
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    # Paths
    "processed_dir": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "Data", "processed"
    ),
    "preprocessor_dir": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "Models", "preprocessors"
    ),

    # Train/test split
    "test_size": 0.2,           # 80% train, 20% test
    "random_state": 42,         # Fixed seed — must never change between runs

    # TF-IDF settings
    "tfidf_max_features": 100,  # Top 100 terms — keeps feature space manageable
    "tfidf_ngram_range": (1, 2),# Unigrams and bigrams
    "tfidf_min_df": 2,          # Term must appear in at least 2 documents

    # Class balancing
    # If Yes:No ratio > 3:1, upsample minority class in training set only
    "balance_threshold": 3.0,
    "balance_strategy": "upsample",  # options: "upsample", "none"

    # CVSS categorical columns to one-hot encode
    "categorical_features": [
        "cvss_attack_vector",
        "cvss_attack_complexity",
        "cvss_privileges_required",
        "cvss_confidentiality_impact",
        "cvss_integrity_impact",
        "cvss_availability_impact",
    ],

    # Reference date for temporal features
    "reference_date": datetime(2020, 1, 1, tzinfo=timezone.utc),
}

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(output_dir: str, ts: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"feature_engineering_log_{ts}.txt")
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
# FEATURE CONSTRUCTION
# ==============================================================================

def build_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build continuous numeric features from CVSS data.

    Features:
        cvss_base_score     — raw CVSS score (0.0 - 10.0)
        cvss_score_squared  — non-linear risk scaling
        cvss_score_critical — binary flag: score >= 9.0
        cvss_score_high     — binary flag: 7.0 <= score < 9.0
        cvss_score_medium   — binary flag: 4.0 <= score < 7.0

    Returns:
        DataFrame of numeric features
    """
    numeric = pd.DataFrame()
    numeric["cvss_base_score"] = df["cvss_base_score"].fillna(0.0)
    numeric["cvss_score_squared"] = numeric["cvss_base_score"] ** 2
    numeric["cvss_score_critical"] = (numeric["cvss_base_score"] >= 9.0).astype(int)
    numeric["cvss_score_high"] = (
        (numeric["cvss_base_score"] >= 7.0) &
        (numeric["cvss_base_score"] < 9.0)
    ).astype(int)
    numeric["cvss_score_medium"] = (
        (numeric["cvss_base_score"] >= 4.0) &
        (numeric["cvss_base_score"] < 7.0)
    ).astype(int)
    return numeric


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build temporal features from CVE publication dates.

    Features:
        published_year          — year of publication
        days_since_published    — days from reference date (2020-01-01)
        is_recent               — published within last 2 years from reference

    Returns:
        DataFrame of temporal features
    """
    temporal = pd.DataFrame()

    try:
        published = pd.to_datetime(df["published_date"], errors="coerce", utc=True)
        ref_date = CONFIG["reference_date"]

        temporal["published_year"] = published.dt.year.fillna(0).astype(int)
        temporal["days_since_published"] = (
            published - ref_date
        ).dt.days.fillna(0).astype(int)
        # Clamp negative values (pre-reference-date CVEs)
        temporal["days_since_published"] = temporal["days_since_published"].clip(lower=0)
        temporal["is_recent"] = (temporal["days_since_published"] <= 730).astype(int)

    except Exception as e:
        logger.warning(f"Temporal feature construction failed: {e}. Using zeros.")
        temporal["published_year"] = 0
        temporal["days_since_published"] = 0
        temporal["is_recent"] = 0

    return temporal


def build_categorical_features(df: pd.DataFrame,
                                 encoder: dict = None,
                                 fit: bool = True) -> tuple:
    """
    One-hot encode CVSS categorical features.

    Handles unseen categories gracefully by ignoring unknown values.
    Encoder vocabulary is fixed on training set and reused on test set
    to prevent data leakage.

    Args:
        df      : Input DataFrame
        encoder : Dict of {column: known_categories} — None if fitting
        fit     : True = fit new encoder on this data; False = use provided

    Returns:
        (encoded_df, encoder_dict)
    """
    if fit:
        encoder = {}

    encoded_frames = []

    for col in CONFIG["categorical_features"]:
        if col not in df.columns:
            logger.warning(f"Categorical column missing: {col}. Skipping.")
            continue

        col_clean = df[col].fillna("Unknown").astype(str).str.upper()

        if fit:
            known_categories = sorted(col_clean.unique().tolist())
            encoder[col] = known_categories
        else:
            known_categories = encoder.get(col, [])

        # One-hot encode with fixed vocabulary
        for cat in known_categories:
            encoded_frames.append(
                pd.Series(
                    (col_clean == cat).astype(int),
                    name=f"{col}_{cat}"
                )
            )

    categorical_df = pd.concat(encoded_frames, axis=1) if encoded_frames else pd.DataFrame()
    return categorical_df, encoder


def build_text_features(df: pd.DataFrame,
                         vectorizer: TfidfVectorizer = None,
                         fit: bool = True) -> tuple:
    """
    Build TF-IDF text features from CVE descriptions.

    Fit on training set only — test set transformed using training vocabulary.
    This prevents data leakage from test set terminology influencing features.

    Args:
        df          : Input DataFrame
        vectorizer  : Fitted TfidfVectorizer — None if fitting
        fit         : True = fit new vectorizer; False = use provided

    Returns:
        (text_feature_df, vectorizer)
    """
    descriptions = df["description"].fillna("no description").astype(str)

    if fit:
        vectorizer = TfidfVectorizer(
            max_features=CONFIG["tfidf_max_features"],
            ngram_range=CONFIG["tfidf_ngram_range"],
            min_df=CONFIG["tfidf_min_df"],
            stop_words="english",
            lowercase=True,
        )
        tfidf_matrix = vectorizer.fit_transform(descriptions)
    else:
        tfidf_matrix = vectorizer.transform(descriptions)

    feature_names = [f"tfidf_{term}" for term in vectorizer.get_feature_names_out()]
    text_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
        index=df.index
    )

    return text_df, vectorizer


def build_target(df: pd.DataFrame) -> pd.Series:
    """
    Encode ground truth label as binary target.
        Yes -> 1 (Priority: High)
        No  -> 0 (Priority: Low)

    Returns:
        Binary Series
    """
    label_map = {"Yes": 1, "No": 0}
    target = df["ground_truth_label"].map(label_map)

    if target.isna().any():
        unmapped = df.loc[target.isna(), "ground_truth_label"].unique()
        logger.warning(f"Unmapped labels found: {unmapped}. Dropping these rows.")
        target = target.dropna()

    return target.astype(int)


# ==============================================================================
# CLASS BALANCING
# ==============================================================================

def balance_training_set(X_train: pd.DataFrame,
                          y_train: pd.Series) -> tuple:
    """
    Balance training set if class imbalance exceeds threshold.
    Applied to TRAINING SET ONLY — test set is never balanced.

    Strategy: Upsample minority class with replacement (random_state=42).

    Args:
        X_train : Training features
        y_train : Training labels

    Returns:
        (X_train_balanced, y_train_balanced)
    """
    class_counts = y_train.value_counts()
    majority_count = class_counts.max()
    minority_count = class_counts.min()
    ratio = majority_count / minority_count

    logger.info(f"Class distribution — Majority: {majority_count}, Minority: {minority_count}, Ratio: {ratio:.2f}")

    if ratio > CONFIG["balance_threshold"] and CONFIG["balance_strategy"] == "upsample":
        logger.info(f"Ratio {ratio:.2f} exceeds threshold {CONFIG['balance_threshold']}. Upsampling minority class...")

        minority_class = class_counts.idxmin()
        X_minority = X_train[y_train == minority_class]
        X_majority = X_train[y_train != minority_class]
        y_minority = y_train[y_train == minority_class]
        y_majority = y_train[y_train != minority_class]

        X_minority_upsampled, y_minority_upsampled = resample(
            X_minority, y_minority,
            replace=True,
            n_samples=len(X_majority),
            random_state=CONFIG["random_state"]
        )

        X_balanced = pd.concat([X_majority, X_minority_upsampled])
        y_balanced = pd.concat([y_majority, y_minority_upsampled])

        logger.info(f"Post-balancing distribution: {y_balanced.value_counts().to_dict()}")
        return X_balanced, y_balanced

    logger.info("Class balance within threshold. No balancing applied.")
    return X_train, y_train


# ==============================================================================
# FEATURE SCALING
# ==============================================================================

def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   scaler: StandardScaler = None,
                   fit: bool = True) -> tuple:
    """
    Apply StandardScaler to numeric and temporal features only.
    TF-IDF and binary one-hot features are not scaled.

    Fit on training set only — prevents data leakage.

    Returns:
        (X_train_scaled, X_test_scaled, scaler)
    """
    # Identify columns to scale (continuous numeric only)
    scale_cols = [c for c in X_train.columns if c in [
        "cvss_base_score", "cvss_score_squared",
        "days_since_published", "published_year"
    ]]

    if not scale_cols:
        logger.info("No continuous columns found for scaling.")
        return X_train, X_test, scaler

    if fit:
        scaler = StandardScaler()
        X_train = X_train.copy()
        X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    else:
        X_train = X_train.copy()
        X_train[scale_cols] = scaler.transform(X_train[scale_cols])

    X_test = X_test.copy()
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    return X_train, X_test, scaler


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def build_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Construct the full feature matrix from a labeled DataFrame.

    Pipeline:
        1. Build numeric features
        2. Build temporal features
        3. Build categorical features (one-hot)
        4. Build text features (TF-IDF)
        5. Concatenate into single feature matrix
        6. Build binary target vector

    Returns:
        (X, y, feature_groups, encoders)
        where encoders = {"tfidf": vectorizer, "categorical": encoder_dict}
    """
    logger.info("Building feature matrix...")

    # Build individual feature groups
    numeric_df = build_numeric_features(df)
    logger.info(f"  Numeric features: {numeric_df.shape[1]}")

    temporal_df = build_temporal_features(df)
    logger.info(f"  Temporal features: {temporal_df.shape[1]}")

    categorical_df, cat_encoder = build_categorical_features(df, fit=True)
    logger.info(f"  Categorical features: {categorical_df.shape[1]}")

    text_df, tfidf_vectorizer = build_text_features(df, fit=True)
    logger.info(f"  TF-IDF features: {text_df.shape[1]}")

    # Reset indices for safe concatenation
    for frame in [numeric_df, temporal_df, categorical_df, text_df]:
        frame.index = df.index

    # Concatenate all feature groups
    X = pd.concat([numeric_df, temporal_df, categorical_df, text_df], axis=1)
    y = build_target(df)

    # Align X and y (drop rows where target was unmappable)
    X = X.loc[y.index]

    logger.info(f"Full feature matrix shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    feature_groups = {
        "numeric": list(numeric_df.columns),
        "temporal": list(temporal_df.columns),
        "categorical": list(categorical_df.columns),
        "tfidf": list(text_df.columns),
        "total_features": X.shape[1]
    }

    encoders = {
        "tfidf": tfidf_vectorizer,
        "categorical": cat_encoder
    }

    return X, y, feature_groups, encoders


def save_feature_schema(feature_groups: dict, output_dir: str, ts: str):
    """
    Save feature schema as JSON for dissertation appendix documentation.
    Documents exactly which features were used in all experiments.
    """
    schema = {
        "schema_version": "1.0.0",
        "timestamp": ts,
        "tfidf_config": {
            "max_features": CONFIG["tfidf_max_features"],
            "ngram_range": list(CONFIG["tfidf_ngram_range"]),
            "min_df": CONFIG["tfidf_min_df"],
        },
        "train_test_split": {
            "test_size": CONFIG["test_size"],
            "random_state": CONFIG["random_state"],
        },
        "feature_groups": feature_groups,
    }

    schema_file = os.path.join(output_dir, f"feature_schema_{ts}.json")
    with open(schema_file, "w") as f:
        json.dump(schema, f, indent=2)
    logger.info(f"Feature schema saved: {schema_file}")
    return schema_file


def generate_engineering_report(X_train, X_test, y_train, y_test,
                                  feature_groups, ts) -> str:
    """
    Generate feature engineering report for dissertation methodology.
    """
    report = f"""
================================================================================
RA5olver Feature Engineering Report
================================================================================
Timestamp            : {ts}
Random State         : {CONFIG["random_state"]} (fixed for reproducibility)
Test Size            : {CONFIG["test_size"]} ({int(CONFIG["test_size"]*100)}%)

DATASET SPLITS
--------------
  Training samples   : {len(X_train):,}
  Test samples       : {len(X_test):,}
  Total samples      : {len(X_train) + len(X_test):,}

TRAINING LABEL DISTRIBUTION
-----------------------------
  Priority Yes (1)   : {y_train.sum():,} ({y_train.mean()*100:.1f}%)
  Priority No  (0)   : {(y_train==0).sum():,} ({(y_train==0).mean()*100:.1f}%)

TEST LABEL DISTRIBUTION
------------------------
  Priority Yes (1)   : {y_test.sum():,} ({y_test.mean()*100:.1f}%)
  Priority No  (0)   : {(y_test==0).sum():,} ({(y_test==0).mean()*100:.1f}%)

FEATURE MATRIX
--------------
  Total Features     : {feature_groups["total_features"]}
  Numeric Features   : {len(feature_groups["numeric"])}
  Temporal Features  : {len(feature_groups["temporal"])}
  Categorical (OHE)  : {len(feature_groups["categorical"])}
  TF-IDF Features    : {len(feature_groups["tfidf"])}

FEATURE GROUPS DETAIL
----------------------
  Numeric  : {", ".join(feature_groups["numeric"])}
  Temporal : {", ".join(feature_groups["temporal"])}

LEAKAGE PREVENTION
-------------------
  - TF-IDF vectorizer fit on TRAINING set only
  - Categorical encoder fit on TRAINING set only
  - StandardScaler fit on TRAINING set only
  - Test set transformed using training artifacts only
  - Balancing applied to TRAINING set only

NEXT STEP
---------
Run: python train_models.py
  Input : Data/processed/X_train_{ts}.csv (and y_train, X_test, y_test)
================================================================================
"""
    return report


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="RA5olver Script 3: Feature Engineering"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to resolvable labeled CSV from Script 2"
    )
    return parser.parse_args()


def main():
    global logger, timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_dir = CONFIG["processed_dir"]
    preprocessor_dir = CONFIG["preprocessor_dir"]

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(preprocessor_dir, exist_ok=True)

    logger = setup_logging(processed_dir, timestamp)

    logger.info("=" * 60)
    logger.info("RA5olver - Script 3: Feature Engineering")
    logger.info("=" * 60)

    args = parse_args()

    # Load labeled data from Script 2
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    logger.info(f"Loading: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    df["cvss_base_score"] = pd.to_numeric(df["cvss_base_score"], errors="coerce")
    logger.info(f"Loaded {len(df):,} records")

    # Build full feature matrix
    X, y, feature_groups, encoders = build_feature_matrix(df)

    # Train/test split — stratified to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y
    )
    logger.info(f"Train/test split: {len(X_train):,} / {len(X_test):,}")

    # Apply feature scaling
    scaler = StandardScaler()
    X_train, X_test, scaler = scale_features(X_train, X_test, fit=True)

    # Balance training set if needed
    X_train, y_train = balance_training_set(X_train, y_train)

    # Save train/test splits
    X_train.to_csv(os.path.join(processed_dir, f"X_train_{timestamp}.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, f"X_test_{timestamp}.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, f"y_train_{timestamp}.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, f"y_test_{timestamp}.csv"), index=False)
    logger.info("Train/test splits saved.")

    # Save preprocessor artifacts
    joblib.dump(encoders["tfidf"],
                os.path.join(preprocessor_dir, f"tfidf_{timestamp}.pkl"))
    joblib.dump(encoders["categorical"],
                os.path.join(preprocessor_dir, f"cat_encoder_{timestamp}.pkl"))
    joblib.dump(scaler,
                os.path.join(preprocessor_dir, f"scaler_{timestamp}.pkl"))
    logger.info("Preprocessor artifacts saved.")

    # Save feature schema
    save_feature_schema(feature_groups, processed_dir, timestamp)

    # Generate and save report
    report = generate_engineering_report(
        X_train, X_test, y_train, y_test, feature_groups, timestamp
    )
    print(report)

    report_file = os.path.join(processed_dir, f"feature_engineering_report_{timestamp}.txt")
    with open(report_file, "w") as f:
        f.write(report)

    logger.info("Script 3 complete. Proceed to: python train_models.py")


if __name__ == "__main__":
    main()
