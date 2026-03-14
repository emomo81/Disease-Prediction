"""
Disease Prediction Flask Webapp
================================
Serves a web interface where users select symptoms and get
top-3 disease predictions with confidence scores and descriptions.
"""

import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# ── Load model artifacts ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model", "disease_model.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "model", "label_encoder.pkl"))
symptom_columns = joblib.load(os.path.join(BASE_DIR, "model", "symptom_columns.pkl"))

# Load disease descriptions
desc_df = pd.read_csv(os.path.join(BASE_DIR, "symptom_Description_820_verified_v5.csv"))
desc_df.columns = desc_df.columns.str.strip()
# Build a dict: disease_name -> description
disease_descriptions = {}
for _, row in desc_df.iterrows():
    name = str(row.iloc[0]).strip().lower()
    desc = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else "No description available."
    disease_descriptions[name] = desc

# Clean symptom names for display
def format_symptom(s):
    """Convert column name to human-readable label."""
    return s.replace('_', ' ').replace('.', ' ').strip().title()

symptom_display = [
    {"value": s, "label": format_symptom(s)}
    for s in symptom_columns
]
# Sort alphabetically for easier browsing
symptom_display.sort(key=lambda x: x["label"])


@app.route("/")
def index():
    return render_template("index.html", symptoms=symptom_display)


@app.route("/predict", methods=["POST"])
def predict():
    selected = request.form.getlist("symptoms")

    if not selected:
        return render_template(
            "index.html",
            symptoms=symptom_display,
            error="Please select at least one symptom."
        )

    # Build binary feature vector
    feature_vector = np.zeros(len(symptom_columns), dtype=np.int8)
    for i, col in enumerate(symptom_columns):
        if col in selected:
            feature_vector[i] = 1

    # Predict probabilities
    input_df = pd.DataFrame([feature_vector], columns=symptom_columns)
    probabilities = model.predict_proba(input_df)[0]

    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]
    results = []
    for idx in top_indices:
        disease_name = label_encoder.inverse_transform([idx])[0]
        confidence = probabilities[idx] * 100
        desc_key = disease_name.strip().lower()
        description = disease_descriptions.get(
            desc_key, "No description available for this condition."
        )
        results.append({
            "name": disease_name.title(),
            "confidence": round(confidence, 2),
            "description": description,
        })

    selected_labels = [format_symptom(s) for s in selected]

    return render_template(
        "result.html",
        results=results,
        selected_symptoms=selected_labels,
    )


if __name__ == "__main__":
    print("\n  🏥  Disease Prediction Webapp")
    print("  ➜  http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)
