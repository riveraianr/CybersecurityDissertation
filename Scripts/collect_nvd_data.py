"""
RA5olver - NVD Data Collection
================================
Script 1 of 6: Research-Grade Data Collection

Purpose:
    Fetches CVE records from the NIST National Vulnerability Database (NVD)
    public API (v2.0). This is the sole data source — no web scraping,
    no third-party enrichment — ensuring full reproducibility.

Research Context:
    Dissertation: "Design Science Approach to Explainable AI for Reducing
    Hallucinations in Vulnerability Management"
    Author: Ian Rivera | Colorado Technical University | 2025

Reproducibility:
    - All parameters are defined in config block below
    - NVD API is free, public, and requires no authentication for basic use
    - Output is a versioned, timestamped CSV saved to Data/raw/
    - Re-running produces a new versioned file; original is never overwritten

Usage:
    python collect_nvd_data.py

Output:
    Data/raw/nvd_raw_YYYYMMDD_HHMMSS.csv
    Data/raw/collection_log_YYYYMMDD_HHMMSS.txt

Dependencies:
    pip install requests pandas tqdm
"""

import os
import time
import logging
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timezone

# ==============================================================================
# CONFIGURATION — all parameters defined here for reproducibility
# ==============================================================================

CONFIG = {
    # Target number of CVEs to collect (1000-2000 per dissertation scope)
    "target_count": 2000,

    # NVD API endpoint (public, no auth required for standard rate limits)
    "nvd_api_url": "https://services.nvd.nist.gov/rest/json/cves/2.0",

    # Results per page — NVD max is 2000
    "results_per_page": 2000,

    # Seconds between API requests — NVD requests ~6 sec between calls without API key
    # With free API key (https://nvd.nist.gov/developers/request-an-api-key): 0.6 sec
    # Set to 6.0 for no-key usage; 0.6 if you have a key
    "request_delay": 6.0,

    # Optional: NVD API key (set as environment variable NVD_API_KEY)
    # Leave as None to run without key (slower but free)
    "api_key": os.getenv("NVD_API_KEY", None),

    # CVSS version preference: "v31" > "v30" > "v2"
    "cvss_preference": ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"],

    # Output paths — relative to script location for portability
    "output_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "raw"),
}

# Adjust delay if API key is present
if CONFIG["api_key"]:
    CONFIG["request_delay"] = 0.6

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(CONFIG["output_dir"], exist_ok=True)

log_file = os.path.join(CONFIG["output_dir"], f"collection_log_{timestamp}.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# DATA COLLECTION
# ==============================================================================

def build_headers():
    """Build request headers. Adds API key if available."""
    headers = {
        "User-Agent": "RA5olver-Dissertation-Research/1.0 (Academic; ian.rivera@student.coloradotech.edu)"
    }
    if CONFIG["api_key"]:
        headers["apiKey"] = CONFIG["api_key"]
    return headers


def fetch_nvd_page(start_index: int, results_per_page: int) -> dict:
    """
    Fetch a single page of CVE records from NVD API.

    Args:
        start_index: Pagination offset
        results_per_page: Number of records to request

    Returns:
        Parsed JSON response dict, or empty dict on failure
    """
    params = {
        "startIndex": start_index,
        "resultsPerPage": results_per_page,
    }

    try:
        response = requests.get(
            CONFIG["nvd_api_url"],
            headers=build_headers(),
            params=params,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error at startIndex={start_index}: {e}")
        if response.status_code == 403:
            logger.warning("Rate limited. Waiting 30 seconds...")
            time.sleep(30)
        return {}

    except requests.exceptions.Timeout:
        logger.error(f"Timeout at startIndex={start_index}. Retrying after 10s...")
        time.sleep(10)
        return {}

    except Exception as e:
        logger.error(f"Unexpected error at startIndex={start_index}: {e}")
        return {}


def collect_cves(target_count: int) -> list:
    """
    Collect CVE records from NVD API with pagination.

    Args:
        target_count: Total number of CVEs to collect

    Returns:
        List of raw CVE item dicts from NVD API
    """
    all_items = []
    start_index = 0
    results_per_page = min(CONFIG["results_per_page"], target_count)

    logger.info(f"Starting NVD collection. Target: {target_count} CVEs")
    logger.info(f"API key present: {bool(CONFIG['api_key'])}")
    logger.info(f"Request delay: {CONFIG['request_delay']}s")

    with tqdm(total=target_count, desc="Collecting CVEs", unit="CVE") as pbar:
        while len(all_items) < target_count:
            remaining = target_count - len(all_items)
            page_size = min(results_per_page, remaining)

            data = fetch_nvd_page(start_index, page_size)

            if not data:
                logger.warning(f"Empty response at startIndex={start_index}. Stopping.")
                break

            items = data.get("vulnerabilities", [])
            if not items:
                logger.info("No more vulnerabilities returned. Collection complete.")
                break

            all_items.extend(items)
            fetched = len(items)
            pbar.update(fetched)

            total_available = data.get("totalResults", 0)
            logger.info(
                f"Fetched {fetched} CVEs (total collected: {len(all_items)} / "
                f"available in NVD: {total_available:,})"
            )

            start_index += fetched

            # Respect rate limit
            if len(all_items) < target_count:
                time.sleep(CONFIG["request_delay"])

    logger.info(f"Collection complete. Total CVEs collected: {len(all_items)}")
    return all_items


# ==============================================================================
# DATA PROCESSING
# ==============================================================================

def extract_cvss_metrics(cve: dict) -> dict:
    """
    Extract CVSS metrics from a CVE record.
    Prioritizes V3.1 > V3.0 > V2.0 per CONFIG preference.

    Args:
        cve: Raw CVE dict from NVD API

    Returns:
        Dict with cvss_version, base_score, severity, vector_string
    """
    metrics = cve.get("metrics", {})

    for version_key in CONFIG["cvss_preference"]:
        metric_list = metrics.get(version_key, [])
        if metric_list:
            cvss_data = metric_list[0].get("cvssData", {})
            return {
                "cvss_version": cvss_data.get("version", "Unknown"),
                "cvss_base_score": cvss_data.get("baseScore", None),
                "cvss_severity": cvss_data.get("baseSeverity",
                    # V2 uses different field name
                    metric_list[0].get("baseSeverity", "Unknown")
                ),
                "cvss_vector": cvss_data.get("vectorString", "Unknown"),
                "cvss_attack_vector": cvss_data.get("attackVector",
                    cvss_data.get("accessVector", "Unknown")
                ),
                "cvss_attack_complexity": cvss_data.get("attackComplexity",
                    cvss_data.get("accessComplexity", "Unknown")
                ),
                "cvss_privileges_required": cvss_data.get("privilegesRequired",
                    cvss_data.get("authentication", "Unknown")
                ),
                "cvss_confidentiality_impact": cvss_data.get("confidentialityImpact", "Unknown"),
                "cvss_integrity_impact": cvss_data.get("integrityImpact", "Unknown"),
                "cvss_availability_impact": cvss_data.get("availabilityImpact", "Unknown"),
            }

    # No CVSS data available
    return {
        "cvss_version": "None",
        "cvss_base_score": None,
        "cvss_severity": "Unknown",
        "cvss_vector": "Unknown",
        "cvss_attack_vector": "Unknown",
        "cvss_attack_complexity": "Unknown",
        "cvss_privileges_required": "Unknown",
        "cvss_confidentiality_impact": "Unknown",
        "cvss_integrity_impact": "Unknown",
        "cvss_availability_impact": "Unknown",
    }


def extract_description(cve: dict) -> str:
    """Extract English description from CVE record."""
    descriptions = cve.get("descriptions", [])
    for desc in descriptions:
        if desc.get("lang") == "en":
            return desc.get("value", "No description available").strip()
    return "No description available"


def extract_weaknesses(cve: dict) -> str:
    """Extract CWE IDs from CVE record as pipe-separated string."""
    weaknesses = cve.get("weaknesses", [])
    cwe_ids = []
    for weakness in weaknesses:
        for desc in weakness.get("description", []):
            value = desc.get("value", "")
            if value.startswith("CWE-"):
                cwe_ids.append(value)
    return "|".join(cwe_ids) if cwe_ids else "Unknown"


def process_cve_item(item: dict) -> dict:
    """
    Transform a raw NVD API CVE item into a flat research record.

    Preserves all fields needed for:
    - Feature engineering (Script 3)
    - Ground truth labeling (Script 2)
    - Full reproducibility audit trail

    Args:
        item: Raw CVE item dict from NVD API response

    Returns:
        Flat dict representing one CVE research record
    """
    cve = item.get("cve", {})
    cve_id = cve.get("id", "Unknown")
    cvss = extract_cvss_metrics(cve)

    return {
        # === Identifiers ===
        "cve_id": cve_id,

        # === Temporal fields ===
        "published_date": cve.get("published", ""),
        "last_modified_date": cve.get("lastModified", ""),

        # === Description ===
        "description": extract_description(cve),

        # === CVSS Metrics (all preserved for feature engineering) ===
        "cvss_version": cvss["cvss_version"],
        "cvss_base_score": cvss["cvss_base_score"],
        "cvss_severity": cvss["cvss_severity"],
        "cvss_vector": cvss["cvss_vector"],
        "cvss_attack_vector": cvss["cvss_attack_vector"],
        "cvss_attack_complexity": cvss["cvss_attack_complexity"],
        "cvss_privileges_required": cvss["cvss_privileges_required"],
        "cvss_confidentiality_impact": cvss["cvss_confidentiality_impact"],
        "cvss_integrity_impact": cvss["cvss_integrity_impact"],
        "cvss_availability_impact": cvss["cvss_availability_impact"],

        # === Weakness Classification ===
        "cwe_ids": extract_weaknesses(cve),

        # === Vulnerability Status ===
        "vuln_status": cve.get("vulnStatus", "Unknown"),

        # === Placeholder columns (populated in Script 2: build_ground_truth.py) ===
        "ground_truth_label": None,
        "ground_truth_rule": None,
        "ground_truth_verified": False,
    }


def process_all_items(items: list) -> pd.DataFrame:
    """
    Process all raw CVE items into a research DataFrame.

    Args:
        items: List of raw CVE items from NVD API

    Returns:
        Processed DataFrame
    """
    logger.info(f"Processing {len(items)} CVE records...")
    records = []

    for item in tqdm(items, desc="Processing CVEs", unit="CVE"):
        try:
            record = process_cve_item(item)
            records.append(record)
        except Exception as e:
            cve_id = item.get("cve", {}).get("id", "Unknown")
            logger.warning(f"Failed to process {cve_id}: {e}")
            continue

    df = pd.DataFrame(records)
    logger.info(f"Processing complete. Shape: {df.shape}")
    return df


# ==============================================================================
# DATA QUALITY REPORTING
# ==============================================================================

def generate_collection_report(df: pd.DataFrame) -> str:
    """
    Generate a data quality report for dissertation documentation.

    Args:
        df: Processed CVE DataFrame

    Returns:
        Report string (also logged and saved)
    """
    total = len(df)
    has_cvss = df["cvss_base_score"].notna().sum()
    missing_cvss = total - has_cvss

    severity_dist = df["cvss_severity"].value_counts().to_dict()

    date_range_start = df["published_date"].min()
    date_range_end = df["published_date"].max()

    cvss_versions = df["cvss_version"].value_counts().to_dict()

    report = f"""
================================================================================
RA5olver NVD Data Collection Report
================================================================================
Collection Timestamp : {timestamp}
NVD API Version      : 2.0
Script Version       : collect_nvd_data.py v1.0

DATASET SUMMARY
---------------
Total CVEs Collected : {total:,}
CVEs with CVSS Score : {has_cvss:,} ({has_cvss/total*100:.1f}%)
CVEs missing CVSS    : {missing_cvss:,} ({missing_cvss/total*100:.1f}%)
Date Range           : {date_range_start} to {date_range_end}

CVSS VERSION DISTRIBUTION
--------------------------
{chr(10).join(f"  {k}: {v:,} ({v/total*100:.1f}%)" for k, v in cvss_versions.items())}

SEVERITY DISTRIBUTION
---------------------
{chr(10).join(f"  {k}: {v:,} ({v/total*100:.1f}%)" for k, v in severity_dist.items())}

COLUMNS PRESERVED FOR RESEARCH PIPELINE
-----------------------------------------
{chr(10).join(f"  - {col}" for col in df.columns)}

NEXT STEP
---------
Run: python build_ground_truth.py
  Input : Data/raw/nvd_raw_{timestamp}.csv
  Output: Data/ground_truth/nvd_labeled_{timestamp}.csv
================================================================================
"""
    return report


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    logger.info("=" * 60)
    logger.info("RA5olver - Script 1: NVD Data Collection")
    logger.info("=" * 60)

    # Step 1: Collect raw CVE data
    raw_items = collect_cves(CONFIG["target_count"])

    if not raw_items:
        logger.error("No data collected. Check network connection and NVD API status.")
        return

    # Step 2: Process into research DataFrame
    df = process_all_items(raw_items)

    # Step 3: Save raw output — versioned, never overwritten
    output_file = os.path.join(CONFIG["output_dir"], f"nvd_raw_{timestamp}.csv")
    df.to_csv(output_file, index=False)
    logger.info(f"Raw data saved: {output_file}")

    # Step 4: Generate and save collection report
    report = generate_collection_report(df)
    print(report)

    report_file = os.path.join(CONFIG["output_dir"], f"collection_report_{timestamp}.txt")
    with open(report_file, "w") as f:
        f.write(report)
    logger.info(f"Collection report saved: {report_file}")

    logger.info("Script 1 complete. Proceed to: python build_ground_truth.py")


if __name__ == "__main__":
    main()
