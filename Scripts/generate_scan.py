import pandas as pd
import os
from openai import OpenAI
import random

# RA5olver - Automated RA-5 Vulnerability Prioritization with XAI
# Copyright © 2025 Ian Rivera. All rights reserved.

# Paths
base_path = r"C:\Users\river\Documents\CybersecurityDissertation"
data_dir = os.path.join(base_path, "Data")
input_file = os.path.join(data_dir, "Security_Vulnerabilities.csv")
output_file = os.path.join(data_dir, "scan_results_full_lr.txt")

# Initialize OpenAI client
client = OpenAI(api_key="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")

# Load the dataset
df_full = pd.read_csv(input_file)

# Select 1280 rows
total_rows = 1280
df = df_full.iloc[:total_rows].copy()  # Create a copy to avoid SettingWithCopyWarning

# Mock CVE_IDs (since dataset lacks them)
mock_cves = [f"CVE-2025-{i:04d}" for i in range(1, len(df_full) + 1)]
df.loc[:, "CVE_ID"] = mock_cves[:total_rows]  # Use .loc to safely assign

# Define severity levels and additional factors
severity_levels = ["Low", "Moderate", "High", "Critical"]
exploitability_factors = ["easy to exploit", "requires specific conditions", "difficult to exploit"]
impact_factors = ["affects multiple systems", "affects single system", "minimal impact"]
mitigation_factors = ["has a fix available", "updates available", "compensating controls exist", "no known mitigations"]
known_status = ["known risk", "unknown risk"]

# Generate analysis for each vulnerability
output_data = []
for index, row in df.iterrows():
    cve_id = row["CVE_ID"]
    summary = row.get("Summary", "No summary available")
    cvss_score = row.get("CVSS_Score", "Unknown")

    # Randomly assign factors for richer data
    exploitability = random.choice(exploitability_factors)
    impact = random.choice(impact_factors)
    mitigation = random.choice(mitigation_factors)
    known = random.choice(known_status)

    # Construct the prompt with more details
    prompt = f"""
    You are a cybersecurity expert analyzing a vulnerability. Provide a detailed analysis in the following format:

    1. Reported Severity: [Low/Moderate/High/Critical based on CVSS score or impact]
    2. XAI Rationale for Severity: [Explain why the severity was chosen, considering CVSS score, exploitability, impact, mitigations, and known status]
    3. Analysis of the vulnerability: [Detailed analysis of the vulnerability, including exploitability, impact, known status, and potential consequences]
    4. Priority (Yes/No) with reasoning: [Determine if this should be a priority based on severity, exploitability, impact, and mitigations]
    5. Conclusion summary: [Summarize the findings and risks]
    6. Suggested Remediation Steps: [Provide actionable steps to mitigate the vulnerability]
    7. Shareable Insight for other systems: [Provide a general insight for other systems]

    Vulnerability Details:
    - CVE ID: {cve_id}
    - Summary: {summary}
    - CVSS Score: {cvss_score}
    - Exploitability: {exploitability}
    - Impact: {impact}
    - Mitigation: {mitigation}
    - Known Status: {known}
    """

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a cybersecurity expert providing detailed vulnerability analysis."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )

    analysis = response.choices[0].message.content.strip()
    output_data.append(f"Row {index + 1}: ### Analysis of {cve_id}\nCVE Link: https://cve.mitre.org/cgi-bin/cvename.cgi?name={cve_id}\n{analysis}\n")
    progress = (index + 1) / total_rows * 100
    print(f"Processed {index + 1}/{total_rows} ({progress:.1f}%)", end="\r", flush=True)

print("\nNew file created!", flush=True)
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("\n".join(output_data))