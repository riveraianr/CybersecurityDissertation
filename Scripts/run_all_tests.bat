@echo off
echo Generating and running tests for 20, 40, 80, 160 rows...

for %%n in (20 40 80 160) do (
    echo Creating xai_test_lr_%%n.py...
    (
        echo import pandas as pd
        echo from sklearn.model_selection import train_test_split
        echo from sklearn.feature_extraction.text import TfidfVectorizer
        echo from sklearn.linear_model import LogisticRegression
        echo from sklearn.metrics import accuracy_score
        echo import numpy as np
        echo import os
        echo import joblib
        echo file_path = r"C:\Users\river\Documents\CybersecurityDissertation\Data\scan_results_full_lr.txt"
        echo model_dir = r"C:\Users\river\Documents\CybersecurityDissertation\Models"
        echo os.makedirs(model_dir, exist_ok=True)
        echo data = []
        echo with open(file_path, 'r', encoding='utf-8', errors='replace') ^as f:
        echo     row = {}
        echo     analysis_lines = []
        echo     remediation_lines = []
        echo     for line in f:
        echo         line = line.strip()
        echo         if line.startswith("Row"):
        echo             if row:
        echo                 row["Analysis"] = "\n".join(analysis_lines)
        echo                 row["Suggested Remediation Steps"] = "\n".join(remediation_lines)
        echo                 data.append(row)
        echo             row = {}
        echo             analysis_lines = []
        echo             remediation_lines = []
        echo         elif "Reported Severity:" in line:
        echo             row["Severity"] = line.split(": ")[1]
        echo         elif "XAI Rationale:" in line:
        echo             row["Severity Rationale"] = line.split(": ")[1]
        echo         elif "Analysis" in line:
        echo             analysis_lines = []
        echo         elif "Suggested Remediation Steps" in line:
        echo             remediation_lines = []
        echo         elif "Priority" in line:
        echo             row["Priority"] = next(f).strip().replace("- ", "")
        echo         elif "Conclusion" in line or "Shareable Insight" in line or line.startswith("Row"):
        echo             continue
        echo         elif analysis_lines is not None and line and not any(l in line for l in ["###", "1.", "2.", "3.", "4.", "5.", "6.", "7."]):
        echo             analysis_lines.append(line)
        echo         elif remediation_lines is not None and line and not any(l in line for l in ["###", "1.", "2.", "3.", "4.", "5.", "6.", "7."]):
        echo             remediation_lines.append(line)
        echo     if row:
        echo         row["Analysis"] = "\n".join(analysis_lines)
        echo         row["Suggested Remediation Steps"] = "\n".join(remediation_lines)
        echo         data.append(row)
        echo df = pd.DataFrame(data[:%%n])
        echo severity_map = {"Low": 1, "Moderate": 2, "High": 3, "Critical": 4}
        echo df["Severity"] = df["Severity"].map(severity_map)
        echo df["combined_text"] = df["Severity Rationale"] + " " + df["Analysis"]
        echo X_text = df["combined_text"]
        echo X_severity = df["Severity"].values.reshape(-1, 1)
        echo y = df["Priority"].map({"Yes": 1, "No": 0})
        echo tfidf = TfidfVectorizer(max_features=500)
        echo X_text_tfidf = tfidf.fit_transform(X_text)
        echo X = np.hstack((X_text_tfidf.toarray(), X_severity))
        echo X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        echo model = LogisticRegression(max_iter=1000)
        echo model.fit(X_train, y_train)
        echo y_pred = model.predict(X_test)
        echo accuracy = accuracy_score(y_test, y_pred)
        echo print(f"Accuracy: {accuracy:.2f}")
        echo joblib.dump(model, os.path.join(model_dir, "lr_model_%%n.pkl"))
        echo joblib.dump(tfidf, os.path.join(model_dir, "tfidf_vectorizer_%%n.pkl"))
        echo print(f"Model and vectorizer saved to {model_dir}")
    ) > xai_test_lr_%%n.py

    echo Creating xai_test_rf_%%n.py...
    (
        echo import pandas as pd
        echo from sklearn.model_selection import train_test_split
        echo from sklearn.feature_extraction.text import TfidfVectorizer
        echo from sklearn.ensemble import RandomForestClassifier
        echo from sklearn.metrics import accuracy_score
        echo import numpy as np
        echo import os
        echo import joblib
        echo file_path = r"C:\Users\river\Documents\CybersecurityDissertation\Data\scan_results_full_lr.txt"
        echo model_dir = r"C:\Users\river\Documents\CybersecurityDissertation\Models"
        echo os.makedirs(model_dir, exist_ok=True)
        echo data = []
        echo with open(file_path, 'r', encoding='utf-8', errors='replace') ^as f:
        echo     row = {}
        echo     analysis_lines = []
        echo     remediation_lines = []
        echo     for line in f:
        echo         line = line.strip()
        echo         if line.startswith("Row"):
        echo             if row:
        echo                 row["Analysis"] = "\n".join(analysis_lines)
        echo                 row["Suggested Remediation Steps"] = "\n".join(remediation_lines)
        echo                 data.append(row)
        echo             row = {}
        echo             analysis_lines = []
        echo             remediation_lines = []
        echo         elif "Reported Severity:" in line:
        echo             row["Severity"] = line.split(": ")[1]
        echo         elif "XAI Rationale:" in line:
        echo             row["Severity Rationale"] = line.split(": ")[1]
        echo         elif "Analysis" in line:
        echo             analysis_lines = []
        echo         elif "Suggested Remediation Steps" in line:
        echo             remediation_lines = []
        echo         elif "Priority" in line:
        echo             row["Priority"] = next(f).strip().replace("- ", "")
        echo         elif "Conclusion" in line or "Shareable Insight" in line or line.startswith("Row"):
        echo             continue
        echo         elif analysis_lines is not None and line and not any(l in line for l in ["###", "1.", "2.", "3.", "4.", "5.", "6.", "7."]):
        echo             analysis_lines.append(line)
        echo         elif remediation_lines is not None and line and not any(l in line for l in ["###", "1.", "2.", "3.", "4.", "5.", "6.", "7."]):
        echo             remediation_lines.append(line)
        echo     if row:
        echo         row["Analysis"] = "\n".join(analysis_lines)
        echo         row["Suggested Remediation Steps"] = "\n".join(remediation_lines)
        echo         data.append(row)
        echo df = pd.DataFrame(data[:%%n])
        echo severity_map = {"Low": 1, "Moderate": 2, "High": 3, "Critical": 4}
        echo df["Severity"] = df["Severity"].map(severity_map)
        echo df["combined_text"] = df["Severity Rationale"] + " " + df["Analysis"]
        echo X_text = df["combined_text"]
        echo X_severity = df["Severity"].values.reshape(-1, 1)
        echo y = df["Priority"].map({"Yes": 1, "No": 0})
        echo tfidf = TfidfVectorizer(max_features=500)
        echo X_text_tfidf = tfidf.fit_transform(X_text)
        echo X = np.hstack((X_text_tfidf.toarray(), X_severity))
        echo X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        echo model = RandomForestClassifier(n_estimators=100, random_state=42)
        echo model.fit(X_train, y_train)
        echo y_pred = model.predict(X_test)
        echo accuracy = accuracy_score(y_test, y_pred)
        echo print(f"Accuracy: {accuracy:.2f}")
        echo joblib.dump(model, os.path.join(model_dir, "rf_model_%%n.pkl"))
        echo joblib.dump(tfidf, os.path.join(model_dir, "tfidf_vectorizer_rf_%%n.pkl"))
        echo print(f"Model and vectorizer saved to {model_dir}")
    ) > xai_test_rf_%%n.py

    echo Running xai_test_lr_%%n.py...
    py -3.11 xai_test_lr_%%n.py
    echo Running xai_test_rf_%%n.py...
    py -3.11 xai_test_rf_%%n.py
    echo.
)

echo All tests complete!
pause