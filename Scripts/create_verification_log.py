"""
Creates a pre-populated verification log CSV from the verification sample.
Run this from your Scripts folder.
"""
import pandas as pd
import os
from datetime import datetime

# ================================================================
# CONFIGURATION — update input path if your timestamp differs
# ================================================================
INPUT_FILE = r"..\Data\ground_truth\verification_sample_20260317_135547.csv"
OUTPUT_FILE = r"..\Data\ground_truth\verification_log_20260317.csv"

def create_verification_log(input_file, output_file):
    print(f"Loading verification sample: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} records")

    # Build verification log
    log = pd.DataFrame()

    # --- Identity columns ---
    log["cve_id"]           = df["cve_id"]
    log["published_date"]   = df["published_date"].str[:10] if "published_date" in df.columns else ""

    # --- CVSS reference columns ---
    log["cvss_base_score"]  = df["cvss_base_score"]
    log["cvss_severity"]    = df["cvss_severity"]
    log["cvss_attack_vector"]     = df.get("cvss_attack_vector", "")
    log["cvss_attack_complexity"] = df.get("cvss_attack_complexity", "")
    log["cvss_privileges_required"] = df.get("cvss_privileges_required", "")
    log["cvss_confidentiality_impact"] = df.get("cvss_confidentiality_impact", "")
    log["cvss_integrity_impact"]    = df.get("cvss_integrity_impact", "")
    log["cvss_availability_impact"] = df.get("cvss_availability_impact", "")

    # --- Description (truncated for readability) ---
    log["description"] = df["description"].str[:300] if "description" in df.columns else ""

    # --- Rule information ---
    log["rule_triggered"]   = df.get("ground_truth_rule_id", "")
    log["rule_name"]        = df.get("ground_truth_rule_name", "")
    log["rule_label"]       = df.get("ground_truth_label", "")
    log["rule_rationale"]   = df.get("ground_truth_rationale", "").str[:200] if "ground_truth_rationale" in df.columns else ""

    # --- Expert verification columns (researcher fills these in) ---
    log["expert_label"]         = ""   # Researcher enters: Yes / No
    log["agreement"]            = ""   # Researcher enters: TRUE / FALSE
    log["override_rationale"]   = ""   # Required if agreement = FALSE
    log["exploitability_notes"] = ""   # Optional: real-world context
    log["confidence"]           = ""   # Optional: High / Medium / Low

    # --- Audit columns ---
    log["verified_by"]        = "Ian Rivera — Cybersecurity Compliance, 10+ years"
    log["verification_date"]  = ""     # Researcher enters date reviewed
    log["record_number"]      = range(1, len(log) + 1)

    # Reorder: put record_number first for easy navigation
    cols = ["record_number", "cve_id", "published_date",
            "cvss_base_score", "cvss_severity",
            "cvss_attack_vector", "cvss_attack_complexity",
            "cvss_privileges_required",
            "cvss_confidentiality_impact", "cvss_integrity_impact",
            "cvss_availability_impact",
            "description",
            "rule_triggered", "rule_name", "rule_label", "rule_rationale",
            "expert_label", "agreement", "override_rationale",
            "exploitability_notes", "confidence",
            "verified_by", "verification_date"]

    log = log[[c for c in cols if c in log.columns]]

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    log.to_csv(output_file, index=False)
    print(f"\nVerification log saved: {output_file}")
    print(f"Records: {len(log)}")
    print(f"\nLabel distribution in sample:")
    print(log["rule_label"].value_counts().to_string())
    print(f"\nColumns in log ({len(log.columns)}):")
    for col in log.columns:
        print(f"  - {col}")
    print(f"\nREADY: Open the CSV, work through records 1-100")
    print(f"Fill in: expert_label, agreement, override_rationale, verification_date")
    print(f"Target: >= 95% agreement rate")

if __name__ == "__main__":
    create_verification_log(INPUT_FILE, OUTPUT_FILE)
