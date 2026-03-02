from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
import openai

# RA5olver - Automated RA-5 Vulnerability Prioritization with XAI
# Copyright © 2025 Ian Rivera. All rights reserved.

app = Flask(__name__, template_folder="../Interface", static_folder="../Interface/assets")

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running the script.")

# Paths
base_path = r"C:\Users\river\Documents\CybersecurityDissertation"
data_dir = os.path.join(base_path, "Data")
output_file = os.path.join(data_dir, "vuln_1280_xai_output.json")
pdp_ice_dir = os.path.join(data_dir, "PDP ICE Output")

# Verify paths exist (optional but helpful for debugging)
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}")
if not os.path.exists(output_file):
    raise FileNotFoundError(f"Vulnerability JSON file not found: {output_file}")

@app.route('/')
def index():
    # Load the output JSON fresh for each request
    try:
        with open(output_file, 'r', encoding='utf-8') as f:  # Added encoding for safety
            vulnerabilities = json.load(f)
        print(f"Loaded {len(vulnerabilities)} vulnerabilities from {output_file}", flush=True)
    except Exception as e:
        print(f"Error loading JSON: {str(e)}", flush=True)
        vulnerabilities = []  # Fallback to empty list if loading fails
    return render_template('index.html', vulnerabilities=vulnerabilities)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get('message', '')
    cve_id = request.form.get('cve_id', '')
    if not user_message or not cve_id:
        return jsonify({"response": "Please provide both a message and a CVE ID."})

    # Load vulnerabilities for consistency
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            vulnerabilities = json.load(f)
    except Exception as e:
        return jsonify({"response": f"Error loading vulnerability data: {str(e)}"})

    vuln = next((v for v in vulnerabilities if v["CVE_ID"] == cve_id), None)
    if not vuln:
        return jsonify({"response": f"Vulnerability {cve_id} not found."})

    prompt = f"""
    You are an expert cybersecurity assistant. A user has asked the following question about {cve_id}:
    "{user_message}"

    Here is the vulnerability data:
    - Analysis: {vuln.get('Analysis', 'Not specified')}
    - Reported Severity: {vuln.get('Reported_Severity', 'Not specified')}
    - Priority: {vuln.get('Priority', 'Not specified')}
    - Predicted Priority: {vuln.get('Predicted_Priority', 'Not specified')}
    - Priority Justification: {vuln.get('Priority_Justification', 'Not specified')}
    - Severity Rationale: {vuln.get('Severity_Rationale', 'Not specified')}
    - Suggested Remediation Steps: {vuln.get('Suggested_Remediation_Steps', 'Not specified')}
    - Shareable Insight: {vuln.get('Shareable_Insight', 'Not specified')}

    Provide a detailed, explainable response to the user's question.
    """
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
        return jsonify({"response": response.choices[0].message.content.strip()})
    except Exception as e:
        return jsonify({"response": f"Error with OpenAI API: {str(e)}"})

@app.route('/pdp_ice/<path:filename>')
def serve_pdp_ice(filename):
    if not os.path.exists(os.path.join(pdp_ice_dir, filename)):
        return jsonify({"error": f"File {filename} not found in PDP ICE directory"}), 404
    return send_from_directory(pdp_ice_dir, filename)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)