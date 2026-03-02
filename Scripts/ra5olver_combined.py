import pandas as pd
import numpy as np
import joblib
import shap
import json
import os
from pdpbox import pdp
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import time
import openai
import plotly.express as px
import plotly.graph_objects as go

# RA5olver - Automated RA-5 Vulnerability Prioritization with XAI
# Copyright © 2025 Ian Rivera. All rights reserved.

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running the script.")

# Paths
base_path = r"C:\Users\river\Documents\CybersecurityDissertation"
data_dir = os.path.join(base_path, "Data")
model_dir = os.path.join(base_path, "Models")
pdp_ice_dir = os.path.join(data_dir, "PDP ICE Output")

# Timestamp for file versioning
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# File paths
input_file = os.path.join(data_dir, "CVE Data", "RA5olver_vulnerability_dataset.csv")
output_file = os.path.join(data_dir, "vuln_1280_xai_output.json")
feedback_file = os.path.join(data_dir, "feedback_data.json")
pdp_html_file = os.path.join(pdp_ice_dir, f"pdp_severity_{timestamp}.html")
ice_html_file = os.path.join(pdp_ice_dir, f"ice_severity_{timestamp}.html")

# Ensure PDP ICE Output directory exists
if not os.path.exists(pdp_ice_dir):
    os.makedirs(pdp_ice_dir)

# Load RF model and TF-IDF vectorizer
rf_model = joblib.load(os.path.join(model_dir, "rf_model_160.pkl"))
tfidf = joblib.load(os.path.join(model_dir, "tfidf_vectorizer_rf_160.pkl"))

# Load feedback data (for reinforcement learning)
if os.path.exists(feedback_file):
    with open(feedback_file, 'r') as f:
        feedback_data = json.load(f)
else:
    feedback_data = {}

# Web search function to fetch additional data (e.g., from Exploit-DB)
def fetch_exploit_data(cve_id):
    try:
        url = f"https://www.exploit-db.com/search?cve={cve_id}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        exploit_exists = bool(soup.find("table", class_="table"))  # Check if exploit table exists
        return {"has_exploit": exploit_exists}
    except Exception as e:
        print(f"Error fetching exploit data for {cve_id}: {e}")
        return {"has_exploit": False}

# OpenAI API function to generate text (updated for new API)
def generate_with_llm(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert cybersecurity assistant helping with vulnerability analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with OpenAI API for prompt '{prompt[:50]}...': {str(e)}")
        return "Not specified"

# Load the new dataset
df = pd.read_csv(input_file)

# Limit to 1280 rows (or fewer if the dataset is smaller)
total_rows = min(1280, len(df))
df = df.iloc[:total_rows]

# Map Reported Severity to numeric (handle 'Unknown')
severity_map = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4, "Unknown": 2}  # Default 'Unknown' to 'Medium'
df["Reported Severity"] = df["Reported_Severity"].map(severity_map)
df = df.dropna(subset=["Reported Severity", "Priority"])

# Fetch additional web data with progress bar
has_exploit_list = []
for cve_id in tqdm(df["CVE_ID"], desc="Fetching Exploit Data"):
    exploit_data = fetch_exploit_data(cve_id)
    has_exploit_list.append(exploit_data["has_exploit"])
    time.sleep(1)  # Avoid rate-limiting
df["Has Exploit"] = has_exploit_list

# Features for prediction with validation (exclude Has Exploit to match model)
df["Analysis"] = df["Analysis"].fillna("Not specified")
df["combined_text"] = df["Analysis"]
# Ensure combined_text has valid strings
df["combined_text"] = df["combined_text"].apply(lambda x: x if isinstance(x, str) and x.strip() != "" else "Default text for TF-IDF")

X_text = tfidf.transform(df["combined_text"]).toarray()
X_severity = df["Reported Severity"].values.reshape(-1, 1)
X = np.hstack((X_text, X_severity))  # Exclude Has Exploit to match the 501 features expected by the model

# Predict with RF (for Priority)
predictions = rf_model.predict(X)
probabilities = rf_model.predict_proba(X)

# Apply reinforcement learning (simulated feedback for now)
for i in range(len(df)):
    cve_id = df.iloc[i]["CVE_ID"]
    predicted_priority = "Yes" if predictions[i] == 1 else "No"
    # Simulate feedback: assume high-severity vulnerabilities with exploits are correctly prioritized as "Yes"
    actual_priority = "Yes" if (df.iloc[i]["Reported Severity"] >= 3 and df.iloc[i]["Has Exploit"]) else "No"
    if cve_id in feedback_data:
        feedback_data[cve_id]["count"] += 1
        feedback_data[cve_id]["correct"] += 1 if predicted_priority == actual_priority else 0
    else:
        feedback_data[cve_id] = {"count": 1, "correct": 1 if predicted_priority == actual_priority else 0}

# Save feedback data
with open(feedback_file, "w") as f:
    json.dump(feedback_data, f, indent=2)

# Adjust predictions based on feedback (simplified RL)
adjusted_predictions = predictions.copy()
for i in range(len(df)):
    cve_id = df.iloc[i]["CVE_ID"]
    if cve_id in feedback_data and feedback_data[cve_id]["count"] > 2:
        accuracy = feedback_data[cve_id]["correct"] / feedback_data[cve_id]["count"]
        if accuracy < 0.5:  # If model is often wrong, adjust prediction
            adjusted_predictions[i] = 1 - predictions[i]  # Flip the prediction

# RF Feature Importance (global)
feature_names = list(tfidf.get_feature_names_out()) + ["Reported Severity"]
rf_importance = dict(zip(feature_names, rf_model.feature_importances_))
top_rf = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:10]
rf_explain = {k: f"{k} drives {v*100:.1f}% of the prediction" for k, v in top_rf}

# SHAP Explainer (for Priority)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)

# Generate Priority Justification with progress bar (using LLM)
priority_justifications = []
for i in tqdm(range(len(df)), desc="Generating Priority Justifications"):
    shap_dict = dict(zip(feature_names, shap_values[i, :, adjusted_predictions[i]]))
    top_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    justification = []
    pred = "Yes" if adjusted_predictions[i] == 1 else "No"
    scan_priority = df.iloc[i]["Priority"]
    justification.append(f"RF model predicted '{pred}' with {probabilities[i][1]*100:.1f}% confidence.")
    if pred != scan_priority:
        justification.append(f"This differs from the scan's priority ('{scan_priority}') due to the following factors:")
        for feature, value in top_shap:
            impact = "strongly" if abs(value) > 0.01 else "moderately" if abs(value) > 0.005 else "slightly"
            direction = "supported" if value > 0 else "opposed"
            justification.append(f"- {feature} {impact} {direction} the prediction ({value:.4f})")
    else:
        justification.append(f"This matches the scan's priority ('{scan_priority}'). Top contributing factors:")
        for feature, value in top_shap:
            impact = "strongly" if abs(value) > 0.01 else "moderately" if abs(value) > 0.005 else "slightly"
            direction = "supported" if value > 0 else "opposed"
            justification.append(f"- {feature} {impact} {direction} the prediction ({value:.4f})")
    # Enhance with LLM
    llm_justification = generate_with_llm(f"Generate a detailed priority justification for {df.iloc[i]['CVE_ID']} with severity {df.iloc[i]['Reported_Severity']} and analysis: {df.iloc[i]['Analysis']}")
    justification.append(f"LLM Insight: {llm_justification}")
    priority_justifications.append("\n".join(justification))

# Enhanced Severity Determination (1-100% scale with XAI) with progress bar (using LLM)
severity_scores = []
severity_rationales = []
for i in tqdm(range(len(df)), desc="Determining Severity"):
    vuln_text = df.iloc[i]["combined_text"].lower()
    has_exploit = df.iloc[i]["Has Exploit"]
    # Base risk score from RF probability (0-1, scaled to 0-100%)
    base_risk = probabilities[i][1] * 100  # Use the "Yes" probability as a starting point
    rationale = ["Base risk score from RF probability: {:.1f}%".format(base_risk)]

    # Adjust based on risk factors
    # Increase risk
    if "multiple systems" in vuln_text or "across systems" in vuln_text:
        base_risk += 20
        rationale.append("Threat to multiple systems: +20%")
    if "easy to exploit" in vuln_text or "easily exploited" in vuln_text:
        base_risk += 15
        rationale.append("Easy to exploit: +15%")
    if "known risk" in vuln_text or "known issue" in vuln_text or "known vulnerability" in vuln_text:
        base_risk += 10
        rationale.append("Known risk: +10%")
    if has_exploit:
        base_risk += 15
        rationale.append("Known exploit exists: +15%")

    # Decrease risk
    if "has a fix" in vuln_text or "fix available" in vuln_text:
        base_risk -= 10
        rationale.append("Has a fix: -10%")
    if "update" in vuln_text and "available" in vuln_text:
        base_risk -= 10
        rationale.append("Updates available: -10%")
    if "compensating controls" in vuln_text or "mitigating controls" in vuln_text:
        base_risk -= 5
        rationale.append("Compensating controls exist: -5%")
    if "unknown risk" in vuln_text or "not widely known" in vuln_text:
        base_risk -= 5
        rationale.append("Unknown risk: -5%")

    # Clamp to 0-100%
    base_risk = max(0, min(100, base_risk))
    rationale.append(f"Final risk score: {base_risk:.1f}%")

    # Map to Critical/High/Moderate/Low
    if base_risk >= 75:
        severity_label = "Critical"
    elif base_risk >= 50:
        severity_label = "High"
    elif base_risk >= 25:
        severity_label = "Moderate"
    else:
        severity_label = "Low"
    rationale.append(f"Severity determined as: {severity_label}")

    # Enhance with LLM
    llm_rationale = generate_with_llm(f"Generate a detailed severity rationale for {df.iloc[i]['CVE_ID']} with severity {df.iloc[i]['Reported_Severity']} and analysis: {df.iloc[i]['Analysis']}")
    rationale.append(f"LLM Insight: {llm_rationale}")

    severity_scores.append(severity_label)
    severity_rationales.append("\n".join(rationale))

# Update DataFrame with new severity and rationale
df["Severity"] = severity_scores
df["Severity Rationale"] = severity_rationales

# Dynamically generate Suggested Remediation Steps and Shareable Insight with LLM
suggested_remediations = []
shareable_insights = []
for i in tqdm(range(len(df)), desc="Generating Remediation and Insights"):
    cve_id = df.iloc[i]["CVE_ID"]
    suggested_remediation = generate_with_llm(f"Generate detailed, step-by-step remediation steps for {cve_id} with analysis: {df.iloc[i]['Analysis']}")
    shareable_insight = generate_with_llm(f"Generate a shareable insight for {cve_id} with analysis: {df.iloc[i]['Analysis']}")
    suggested_remediations.append(suggested_remediation)
    shareable_insights.append(shareable_insight)

df["Suggested_Remediation_Steps"] = suggested_remediations
df["Shareable_Insight"] = shareable_insights

# PDP/ICE for Reported Severity (using Plotly for interactivity)
pdp_severity = pdp.pdp_isolate(model=rf_model, dataset=pd.DataFrame(X, columns=feature_names), 
                               model_features=feature_names, feature="Reported Severity")
fig_pdp = go.Figure()
fig_pdp.add_trace(go.Scatter(x=pdp_severity.feature_grids, y=pdp_severity.pdp, mode='lines', name='PDP'))
fig_pdp.update_layout(title="Partial Dependence Plot for Reported Severity", xaxis_title="Reported Severity", yaxis_title="Predicted Priority")
fig_pdp.write_html(pdp_html_file)

pdp_ice = pdp.pdp_isolate(model=rf_model, dataset=pd.DataFrame(X[:100], columns=feature_names), 
                          model_features=feature_names, feature="Reported Severity")
fig_ice = go.Figure()
for ice_line in pdp_ice.ice_lines.values:
    fig_ice.add_trace(go.Scatter(x=pdp_ice.feature_grids, y=ice_line, mode='lines', opacity=0.1, line=dict(color='blue')))
fig_ice.add_trace(go.Scatter(x=pdp_ice.feature_grids, y=pdp_ice.pdp, mode='lines', name='PDP', line=dict(color='red', width=2)))
fig_ice.update_layout(title="ICE Plot for Reported Severity", xaxis_title="Reported Severity", yaxis_title="Predicted Priority")
fig_ice.write_html(ice_html_file)

# Output for all rows with progress bar
outputs = []
for i in tqdm(range(len(df)), desc="Generating Output"):
    vuln_data = df.iloc[i].to_dict()
    pred = "Yes" if adjusted_predictions[i] == 1 else "No"
    vuln_data["Predicted_Priority"] = pred
    vuln_data["Priority_Confidence"] = float(max(probabilities[i]))
    vuln_data["Priority Justification"] = priority_justifications[i]
    vuln_data["RF_Feature_Importance"] = rf_explain
    shap_dict = dict(zip(feature_names, shap_values[i, :, adjusted_predictions[i]]))
    top_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    shap_explain = {}
    for k, v in top_shap:
        impact = "strongly" if abs(v) > 0.01 else "moderately" if abs(v) > 0.005 else "slightly"
        direction = "supported" if v > 0 else "opposed"
        shap_explain[k] = f"{k} {impact} {direction} {pred} ({v:.4f})"
    vuln_data["SHAP_Values"] = shap_explain
    vuln_data["PDP_ICE"] = {"PDP": f"pdp_severity_{timestamp}.html", "ICE": f"ice_severity_{timestamp}.html"}
    outputs.append(vuln_data)

# Save as JSON
with open(output_file, "w") as f:
    json.dump(outputs, f, indent=2)
print(f"Processed {len(df)} vulnerabilities - output saved to {output_file}")