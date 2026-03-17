"""
RA5olver - Ground Truth Labeling Engine
========================================
Script 2 of 6: Expert-Validated Ground Truth Construction

Purpose:
    Applies a transparent, rule-based labeling system to the raw NVD dataset
    to produce ground truth labels for the experimental comparison.
    Labels define what a "correct" priority prediction looks like — this is
    the benchmark all 6 experimental conditions are measured against.

Research Context:
    Dissertation: "Design Science Approach to Explainable AI for Reducing
    Hallucinations in Vulnerability Management"
    Author: Ian Rivera | Colorado Technical University | 2025

Methodology Note:
    Ground truth rules are derived from:
    1. NIST CVSS scoring thresholds (industry standard, citable)
    2. NIST SP 800-53 RA-5 remediation priority guidance
    3. Expert validation by the researcher (10+ years cybersecurity compliance)

    Rules are fully documented and static — they do not change between
    experimental runs. This ensures consistent, reproducible labeling.

    Hallucination Definition (Operationalized):
        A model output is classified as a hallucination when its predicted
        priority label differs from the expert-validated ground truth label
        for a given CVE record.

Reproducibility:
    - Rules are version-controlled in GROUND_TRUTH_RULES dict below
    - All labeling decisions are logged with the rule that triggered them
    - Output includes verification flags for manual audit sampling

Usage:
    python build_ground_truth.py --input Data/raw/nvd_raw_YYYYMMDD_HHMMSS.csv

Output:
    Data/ground_truth/nvd_labeled_YYYYMMDD_HHMMSS.csv
    Data/ground_truth/labeling_report_YYYYMMDD_HHMMSS.txt

Dependencies:
    pip install pandas tqdm
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    "output_dir": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "Data", "ground_truth"
    ),
    "rule_set_version": "1.0.0",  # Increment if rules change — tracks in dissertation
}

# ==============================================================================
# GROUND TRUTH RULE SET
# ==============================================================================
# These rules are the operationalized definition of "Priority: Yes" for this study.
# Derived from NIST CVSS thresholds and RA-5 guidance.
# Each rule has:
#   - id        : unique identifier for traceability
#   - label     : resulting ground truth label
#   - rationale : citation-ready justification for dissertation

GROUND_TRUTH_RULES = {
    "version": CONFIG["rule_set_version"],
    "citation": "NIST SP 800-53 Rev. 5 RA-5; NIST CVSS v3.1 Scoring Guide",

    "rules": [
        {
            "id": "GTR-001",
            "name": "Critical CVSS Score",
            "label": "Yes",
            "priority_tier": 1,
            "condition": lambda row: (
                pd.notna(row["cvss_base_score"]) and
                float(row["cvss_base_score"]) >= 9.0
            ),
            "rationale": (
                "CVSS base score >= 9.0 is classified as Critical per NIST CVSS v3.1 "
                "severity ratings. RA-5 mandates immediate remediation for critical "
                "vulnerabilities. (NIST, 2020)"
            )
        },
        {
            "id": "GTR-002",
            "name": "High CVSS Score with Network Attack Vector",
            "label": "Yes",
            "priority_tier": 1,
            "condition": lambda row: (
                pd.notna(row["cvss_base_score"]) and
                float(row["cvss_base_score"]) >= 7.0 and
                str(row["cvss_attack_vector"]).upper() in ["NETWORK", "AV:N"]
            ),
            "rationale": (
                "CVSS base score >= 7.0 (High) with network-based attack vector "
                "indicates remotely exploitable vulnerability. Network-accessible "
                "vulnerabilities represent elevated organizational risk per RA-5 "
                "continuous monitoring requirements. (NIST, 2020)"
            )
        },
        {
            "id": "GTR-003",
            "name": "High CVSS Score with Low Attack Complexity",
            "label": "Yes",
            "priority_tier": 1,
            "condition": lambda row: (
                pd.notna(row["cvss_base_score"]) and
                float(row["cvss_base_score"]) >= 7.0 and
                str(row["cvss_attack_complexity"]).upper() in ["LOW", "AC:L"]
            ),
            "rationale": (
                "CVSS base score >= 7.0 (High) with low attack complexity indicates "
                "the vulnerability is easily exploitable without specialized conditions. "
                "Low complexity exploits are prioritized in RA-5 remediation workflows "
                "due to increased likelihood of exploitation. (NIST, 2020)"
            )
        },
        {
            "id": "GTR-004",
            "name": "High CVSS Score with Full CIA Impact",
            "label": "Yes",
            "priority_tier": 1,
            "condition": lambda row: (
                pd.notna(row["cvss_base_score"]) and
                float(row["cvss_base_score"]) >= 7.0 and
                str(row["cvss_confidentiality_impact"]).upper() == "HIGH" and
                str(row["cvss_integrity_impact"]).upper() == "HIGH" and
                str(row["cvss_availability_impact"]).upper() == "HIGH"
            ),
            "rationale": (
                "CVSS base score >= 7.0 with HIGH impact across Confidentiality, "
                "Integrity, and Availability (full CIA triad compromise) represents "
                "maximum potential organizational damage. Such vulnerabilities warrant "
                "immediate prioritization per RA-5 risk assessment criteria. (NIST, 2020)"
            )
        },
        {
            "id": "GTR-005",
            "name": "High CVSS Score Baseline",
            "label": "Yes",
            "priority_tier": 2,
            "condition": lambda row: (
                pd.notna(row["cvss_base_score"]) and
                float(row["cvss_base_score"]) >= 7.0
            ),
            "rationale": (
                "CVSS base score >= 7.0 meets the High severity threshold per NIST "
                "CVSS v3.1 severity ratings. NIST SP 800-53 RA-5 identifies high-severity "
                "vulnerabilities as requiring prioritized remediation. (NIST, 2020)"
            )
        },
        {
            "id": "GTR-006",
            "name": "Medium CVSS Score with No Privileges Required",
            "label": "Yes",
            "priority_tier": 2,
            "condition": lambda row: (
                pd.notna(row["cvss_base_score"]) and
                4.0 <= float(row["cvss_base_score"]) < 7.0 and
                str(row["cvss_privileges_required"]).upper() in ["NONE", "PR:N"]
            ),
            "rationale": (
                "Medium CVSS score (4.0-6.9) with no privileges required indicates "
                "an unauthenticated attack surface. While below High threshold, "
                "zero-privilege medium vulnerabilities represent meaningful exposure "
                "and warrant priority attention per RA-5 risk-based prioritization. "
                "(NIST, 2020)"
            )
        },
        {
            "id": "GTR-007",
            "name": "Low Priority - Low CVSS Score",
            "label": "No",
            "priority_tier": 3,
            "condition": lambda row: (
                pd.notna(row["cvss_base_score"]) and
                float(row["cvss_base_score"]) < 4.0
            ),
            "rationale": (
                "CVSS base score < 4.0 falls below NIST Low/Medium threshold. "
                "While not zero risk, low-severity vulnerabilities are deprioritized "
                "relative to High/Critical findings in RA-5 remediation workflows. "
                "(NIST, 2020)"
            )
        },
        {
            "id": "GTR-008",
            "name": "No CVSS Data Available",
            "label": "Unresolvable",
            "priority_tier": 4,
            "condition": lambda row: pd.isna(row["cvss_base_score"]),
            "rationale": (
                "Insufficient CVSS data to apply rule-based labeling. Records with "
                "no CVSS score are excluded from model training and evaluation to "
                "prevent ground truth uncertainty from contaminating results. "
                "Documented as study limitation."
            )
        },
    ]
}

# ==============================================================================
# LABELING ENGINE
# ==============================================================================

def apply_ground_truth_rules(row: pd.Series) -> dict:
    """
    Apply ground truth rules to a single CVE record.
    Rules are evaluated in priority_tier order — first match wins.

    Args:
        row: A single row from the processed NVD DataFrame

    Returns:
        Dict with ground_truth_label, ground_truth_rule_id,
        ground_truth_rule_name, ground_truth_rationale,
        ground_truth_tier
    """
    # Sort rules by priority tier (lower = higher priority)
    sorted_rules = sorted(
        GROUND_TRUTH_RULES["rules"],
        key=lambda r: r["priority_tier"]
    )

    for rule in sorted_rules:
        try:
            if rule["condition"](row):
                return {
                    "ground_truth_label": rule["label"],
                    "ground_truth_rule_id": rule["id"],
                    "ground_truth_rule_name": rule["name"],
                    "ground_truth_rationale": rule["rationale"],
                    "ground_truth_tier": rule["priority_tier"],
                    "ground_truth_rule_version": GROUND_TRUTH_RULES["version"],
                }
        except Exception as e:
            continue

    # Fallback — should not reach here with complete rule set
    return {
        "ground_truth_label": "Unresolvable",
        "ground_truth_rule_id": "GTR-FALLBACK",
        "ground_truth_rule_name": "No rule matched",
        "ground_truth_rationale": "No rule condition was satisfied.",
        "ground_truth_tier": 99,
        "ground_truth_rule_version": GROUND_TRUTH_RULES["version"],
    }


def label_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ground truth labeling to entire dataset.

    Args:
        df: Raw NVD DataFrame from Script 1

    Returns:
        DataFrame with ground truth columns populated
    """
    logger.info("Applying ground truth rules to dataset...")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Labeling CVEs", unit="CVE"):
        result = apply_ground_truth_rules(row)
        results.append(result)

    label_df = pd.DataFrame(results)

    # Merge labels back into main DataFrame
    df = df.copy()
    df["ground_truth_label"] = label_df["ground_truth_label"].values
    df["ground_truth_rule_id"] = label_df["ground_truth_rule_id"].values
    df["ground_truth_rule_name"] = label_df["ground_truth_rule_name"].values
    df["ground_truth_rationale"] = label_df["ground_truth_rationale"].values
    df["ground_truth_tier"] = label_df["ground_truth_tier"].values
    df["ground_truth_rule_version"] = label_df["ground_truth_rule_version"].values

    # Verification flag — marks records for manual audit sampling
    # Default False; researcher sets to True after manual review
    df["ground_truth_verified"] = False

    logger.info("Labeling complete.")
    return df


def filter_resolvable(df: pd.DataFrame) -> tuple:
    """
    Separate resolvable from unresolvable records.
    Unresolvable records (no CVSS) are excluded from experiments
    and documented as a study limitation.

    Returns:
        (resolvable_df, excluded_df)
    """
    resolvable = df[df["ground_truth_label"] != "Unresolvable"].copy()
    excluded = df[df["ground_truth_label"] == "Unresolvable"].copy()

    logger.info(
        f"Resolvable records: {len(resolvable):,} | "
        f"Excluded (no CVSS): {len(excluded):,}"
    )
    return resolvable, excluded


# ==============================================================================
# VERIFICATION SAMPLING
# ==============================================================================

def generate_verification_sample(df: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
    """
    Generate a stratified random sample for manual researcher verification.

    Stratified by:
    - ground_truth_label (Yes / No)
    - ground_truth_tier

    This sample is what the researcher manually reviews to validate
    the labeling system — documented in the dissertation methodology.

    Args:
        df: Labeled DataFrame
        sample_size: Number of records to sample for verification

    Returns:
        Verification sample DataFrame
    """
    try:
        sample = df.groupby("ground_truth_label", group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), sample_size // df["ground_truth_label"].nunique()),
                random_state=42
            )
        )
        logger.info(f"Verification sample generated: {len(sample)} records")
        return sample
    except Exception as e:
        logger.warning(f"Stratified sampling failed, using simple random sample: {e}")
        return df.sample(min(sample_size, len(df)), random_state=42)


# ==============================================================================
# REPORTING
# ==============================================================================

def generate_labeling_report(df_full: pd.DataFrame,
                              df_resolvable: pd.DataFrame,
                              df_excluded: pd.DataFrame,
                              timestamp: str) -> str:
    """
    Generate a labeling report for dissertation methodology documentation.
    """
    total = len(df_full)
    resolvable = len(df_resolvable)
    excluded = len(df_excluded)

    label_dist = df_resolvable["ground_truth_label"].value_counts().to_dict()
    rule_dist = df_resolvable["ground_truth_rule_id"].value_counts().to_dict()
    tier_dist = df_resolvable["ground_truth_tier"].value_counts().sort_index().to_dict()

    yes_count = label_dist.get("Yes", 0)
    no_count = label_dist.get("No", 0)
    balance_ratio = yes_count / no_count if no_count > 0 else float("inf")

    report = f"""
================================================================================
RA5olver Ground Truth Labeling Report
================================================================================
Labeling Timestamp   : {timestamp}
Rule Set Version     : {GROUND_TRUTH_RULES["version"]}
Citation Basis       : {GROUND_TRUTH_RULES["citation"]}

DATASET SUMMARY
---------------
Total CVEs Input     : {total:,}
Resolvable Records   : {resolvable:,} ({resolvable/total*100:.1f}%)
Excluded (no CVSS)   : {excluded:,} ({excluded/total*100:.1f}%)

LABEL DISTRIBUTION (Resolvable Records)
----------------------------------------
  Priority Yes       : {yes_count:,} ({yes_count/resolvable*100:.1f}%)
  Priority No        : {no_count:,} ({no_count/resolvable*100:.1f}%)
  Yes:No Ratio       : {balance_ratio:.2f}

  NOTE: Class imbalance ratio documented here for dissertation.
  If ratio > 3:1, consider stratified sampling in model training (Script 4).

RULE TRIGGER DISTRIBUTION
--------------------------
{chr(10).join(f"  {k}: {v:,} ({v/resolvable*100:.1f}%)" for k, v in rule_dist.items())}

PRIORITY TIER DISTRIBUTION
---------------------------
{chr(10).join(f"  Tier {k}: {v:,} records" for k, v in tier_dist.items())}

GROUND TRUTH RULES APPLIED
---------------------------
{chr(10).join(f"  {r['id']}: {r['name']}" for r in GROUND_TRUTH_RULES["rules"])}

VERIFICATION INSTRUCTIONS
--------------------------
A stratified sample of 100 records has been saved to:
  Data/ground_truth/verification_sample_{timestamp}.csv

Researcher Action Required:
  1. Open verification_sample CSV
  2. Review each record's ground_truth_label against CVE description
  3. Set ground_truth_verified = True for confirmed records
  4. Document any disagreements — these become methodology notes
  5. Target: >= 95% agreement rate to validate rule set

NEXT STEP
---------
Run: python feature_engineering.py
  Input : Data/ground_truth/nvd_labeled_{timestamp}.csv
================================================================================
"""
    return report


# ==============================================================================
# MAIN
# ==============================================================================

def setup_logging(output_dir: str, timestamp: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"labeling_log_{timestamp}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="RA5olver Script 2: Ground Truth Labeling Engine"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw NVD CSV from Script 1 (e.g., Data/raw/nvd_raw_YYYYMMDD.csv)"
    )
    return parser.parse_args()


def main():
    global logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = CONFIG["output_dir"]
    logger = setup_logging(output_dir, timestamp)

    logger.info("=" * 60)
    logger.info("RA5olver - Script 2: Ground Truth Labeling Engine")
    logger.info(f"Rule Set Version: {GROUND_TRUTH_RULES['version']}")
    logger.info("=" * 60)

    args = parse_args()

    # Load raw data from Script 1
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    logger.info(f"Loading input file: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    logger.info(f"Loaded {len(df):,} records")

    # Ensure cvss_base_score is numeric
    df["cvss_base_score"] = pd.to_numeric(df["cvss_base_score"], errors="coerce")

    # Apply ground truth labeling
    df_labeled = label_dataset(df)

    # Separate resolvable from excluded
    df_resolvable, df_excluded = filter_resolvable(df_labeled)

    # Save full labeled dataset
    labeled_file = os.path.join(output_dir, f"nvd_labeled_{timestamp}.csv")
    df_labeled.to_csv(labeled_file, index=False)
    logger.info(f"Full labeled dataset saved: {labeled_file}")

    # Save excluded records separately for documentation
    if len(df_excluded) > 0:
        excluded_file = os.path.join(output_dir, f"nvd_excluded_{timestamp}.csv")
        df_excluded.to_csv(excluded_file, index=False)
        logger.info(f"Excluded records saved: {excluded_file}")

    # Save resolvable dataset (used in all downstream scripts)
    resolvable_file = os.path.join(output_dir, f"nvd_resolvable_{timestamp}.csv")
    df_resolvable.to_csv(resolvable_file, index=False)
    logger.info(f"Resolvable dataset saved: {resolvable_file}")

    # Generate verification sample for manual researcher review
    verification_sample = generate_verification_sample(df_resolvable, sample_size=100)
    sample_file = os.path.join(output_dir, f"verification_sample_{timestamp}.csv")
    verification_sample.to_csv(sample_file, index=False)
    logger.info(f"Verification sample saved: {sample_file}")

    # Generate and save labeling report
    report = generate_labeling_report(
        df_labeled, df_resolvable, df_excluded, timestamp
    )
    print(report)

    report_file = os.path.join(output_dir, f"labeling_report_{timestamp}.txt")
    with open(report_file, "w") as f:
        f.write(report)
    logger.info(f"Labeling report saved: {report_file}")

    logger.info("Script 2 complete. Proceed to: python feature_engineering.py")


if __name__ == "__main__":
    main()
