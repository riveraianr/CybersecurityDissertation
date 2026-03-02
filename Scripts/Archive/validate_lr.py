import joblib
import pandas as pd
import numpy as np

# Load the saved model and vectorizer
model = joblib.load(r"C:\Users\river\Documents\CybersecurityDissertation\Models\lr_model.pkl")
tfidf = joblib.load(r"C:\Users\river\Documents\CybersecurityDissertation\Models\tfidf_vectorizer.pkl")

# Row 0 data from scan_results_full_lr.txt (hardcoded for simplicity)
severity = "Moderate"  # Reported Severity
severity_rationale = "Moderate fits because the vulnerability allows for arbitrary file read, which can potentially lead to sensitive data exposure and unauthorized access. While not as severe as code execution, the impact can still be significant. Adjusting the severity to High could also be justified depending on the specific context and environment."
analysis = "Full analysis: The vulnerability in Ghost allows an attacker to exploit symlinks during content import, leading to arbitrary file read. By crafting malicious symlinks, an attacker can access sensitive information stored on the server, such as configuration files or user data. This could result in data leakage, unauthorized access, or further exploitation of the system."
true_priority = "Yes"  # Ground truth from Row 0

# Map Severity to numeric
severity_map = {"Low": 1, "Moderate": 2, "High": 3, "Critical": 4}
severity_numeric = severity_map[severity]

# Combine text for vectorization
combined_text = severity_rationale + " " + analysis

# Vectorize the text
X_text_tfidf = tfidf.transform([combined_text]).toarray()

# Combine with Severity
X = np.hstack((X_text_tfidf, [[severity_numeric]]))

# Predict
prediction = model.predict(X)[0]
predicted_priority = "Yes" if prediction == 1 else "No"

# Output
print(f"True Priority: {true_priority}")
print(f"Predicted Priority: {predicted_priority}")
print(f"Match: {true_priority == predicted_priority}")