"""
Quick Model Accuracy Checker
Loads the current model and displays its accuracy on test data.
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("  Current Model Accuracy Check")
print("=" * 60)

# Load current model
try:
    model = joblib.load("model/disease_model.pkl")
    print("\n✓ Loaded model from model/disease_model.pkl")
except FileNotFoundError:
    print("\n❌ No model found! Run training first.")
    exit(1)

# Load metadata if exists
try:
    metadata = joblib.load("model/model_metadata.pkl")
    print(f"\n📊 Model Info:")
    print(f"   Algorithm: {metadata['algorithm']}")
    print(f"   Accuracy: {metadata['accuracy']*100:.2f}%")
    print(f"   Macro F1: {metadata['macro_f1']:.4f}")
except FileNotFoundError:
    print("\n⚠️  No metadata found (trained with old script)")

# Load and prepare test data
print("\n🔄 Loading dataset for validation...")
df = pd.read_csv("unified_820_diseases_symptoms_dataset_ROWLEVEL_CLEANED.csv")
df.columns = df.columns.str.strip()

duplicate_cols = [c for c in df.columns if c.endswith('.1')]
if duplicate_cols:
    df.drop(columns=duplicate_cols, inplace=True)

target_col = 'diseases'
symptom_columns = [c for c in df.columns if c != target_col]

# Filter rare diseases
counts = df[target_col].value_counts()
rare = counts[counts < 2].index.tolist()
if rare:
    df = df[~df[target_col].isin(rare)]

X = df[symptom_columns].astype(np.int8)
y = df[target_col]

# Encode
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Predict
print("🔄 Running predictions on test set...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Get detailed metrics
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
macro = report['macro avg']

print("\n" + "=" * 60)
print(f"  📊 Test Set Accuracy: {accuracy*100:.2f}%")
print("=" * 60)
print(f"\nMacro Average Metrics:")
print(f"  Precision: {macro['precision']:.4f}")
print(f"  Recall:    {macro['recall']:.4f}")
print(f"  F1-Score:  {macro['f1-score']:.4f}")
print("\n" + "=" * 60)

# Show class distribution
print(f"\n📈 Dataset Info:")
print(f"   Total samples: {len(df):,}")
print(f"   Test samples: {len(X_test):,}")
print(f"   Number of diseases: {len(le.classes_)}")
print(f"   Number of symptoms: {len(symptom_columns)}")

print("\n" + "=" * 60)
