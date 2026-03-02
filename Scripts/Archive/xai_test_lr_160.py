import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import os
import joblib

# File path
file_path = r"C:\Users\river\Documents\CybersecurityDissertation\Data\scan_results_full_lr.txt"
model_dir = r"C:\Users\river\Documents\CybersecurityDissertation\Models"

# Create Models directory if it doesn’t exist
os.makedirs(model_dir, exist_ok=True)

# Custom parser for the structured text file
data = []
with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    row = {}
    analysis_lines = []
    remediation_lines = []
    for line in f:
        line = line.strip()
        if line.startswith("Row"):
            if row:  # Save previous row
                row["Analysis"] = "\n".join(analysis_lines)
                row["Suggested Remediation Steps"] = "\n".join(remediation_lines)
                if "Priority" not in row or row["Priority"] not in ["Yes", "No"]:
                    row["Priority"] = "No"
                data.append(row)
            row = {}
            analysis_lines = []
            remediation_lines = []
        elif "Reported Severity:" in line:
            row["Severity"] = line.split(": ")[1]
        elif "XAI Rationale:" in line:
            row["Severity Rationale"] = line.split(": ")[1]
        elif "Analysis" in line:
            analysis_lines = []
        elif "Suggested Remediation Steps" in line:
            remediation_lines = []
        elif "Priority" in line:
            next_line = next(f).strip().replace("- ", "")
            row["Priority"] = next_line if next_line in ["Yes", "No"] else "No"
        elif "Conclusion" in line or "Shareable Insight" in line or line.startswith("Row"):
            continue
        elif analysis_lines is not None and line and not any(l in line for l in ["###", "1.", "2.", "3.", "4.", "5.", "6.", "7."]):
            analysis_lines.append(line)
        elif remediation_lines is not None and line and not any(l in line for l in ["###", "1.", "2.", "3.", "4.", "5.", "6.", "7."]):
            remediation_lines.append(line)
    if row:  # Save the last row
        row["Analysis"] = "\n".join(analysis_lines)
        row["Suggested Remediation Steps"] = "\n".join(remediation_lines)
        if "Priority" not in row or row["Priority"] not in ["Yes", "No"]:
            row["Priority"] = "No"
        data.append(row)

# Convert to DataFrame
df = pd.DataFrame(data[:160])

# Map Severity to numeric values
severity_map = {"Low": 1, "Moderate": 2, "High": 3, "Critical": 4}
df["Severity"] = df["Severity"].map(severity_map)
df = df.dropna(subset=["Severity", "Priority"])

# Combine text columns for vectorization
df["combined_text"] = df["Severity Rationale"] + " " + df["Analysis"]

# Features (X) and target (y)
X_text = df["combined_text"]
X_severity = df["Severity"].values.reshape(-1, 1)
y = df["Priority"].map({"Yes": 1, "No": 0})

# Vectorize text data
tfidf = TfidfVectorizer(max_features=500)
X_text_tfidf = tfidf.fit_transform(X_text)

# Combine features
X = np.hstack((X_text_tfidf.toarray(), X_severity))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# Save the model and vectorizer
joblib.dump(model, os.path.join(model_dir, "lr_model_160.pkl"))
joblib.dump(tfidf, os.path.join(model_dir, "tfidf_vectorizer_160.pkl"))
print(f"Model and vectorizer saved to {model_dir}")