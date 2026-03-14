"""
Disease Prediction Model Training Script
=========================================
Trains a Random Forest classifier on the 820-disease symptom dataset
and exports the model artifacts for use in the Flask webapp.
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


def main():
    print("=" * 60)
    print("  Disease Prediction Model — Training Pipeline")
    print("=" * 60)

    # ── 1. Load dataset ──────────────────────────────────────────
    print("\n[1/6] Loading dataset...")
    start = time.time()
    df = pd.read_csv("unified_820_diseases_symptoms_dataset_ROWLEVEL_CLEANED.csv")
    print(f"  ✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns in {time.time()-start:.1f}s")

    # ── 2. Clean columns ─────────────────────────────────────────
    print("\n[2/6] Cleaning columns...")
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Drop duplicate symptom columns (e.g. itching.1, dry skin.1, muscle pain.1)
    duplicate_cols = [c for c in df.columns if c.endswith('.1')]
    if duplicate_cols:
        df.drop(columns=duplicate_cols, inplace=True)
        print(f"  ✓ Dropped {len(duplicate_cols)} duplicate columns")

    # Separate features and target
    target_col = 'diseases'
    symptom_columns = [c for c in df.columns if c != target_col]

    # Filter out diseases with fewer than 2 samples (can't stratify-split them)
    counts = df[target_col].value_counts()
    rare = counts[counts < 2].index.tolist()
    if rare:
        df = df[~df[target_col].isin(rare)]
        print(f"  ✓ Filtered out {len(rare)} diseases with < 2 samples")

    # Convert features to int8 to save memory (values are 0/1)
    X = df[symptom_columns].astype(np.int8)
    y = df[target_col]

    print(f"  ✓ Features: {X.shape[1]} symptoms (int8 = {X.memory_usage(deep=True).sum()/1024/1024:.0f} MB)")
    print(f"  ✓ Target: {y.nunique()} unique diseases ({len(df):,} rows)")

    # ── 3. Encode target ─────────────────────────────────────────
    print("\n[3/6] Encoding target labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"  ✓ Encoded {len(le.classes_)} disease classes")

    # ── 4. Train/test split ───────────────────────────────────────
    print("\n[4/6] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"  ✓ Train: {X_train.shape[0]:,} samples")
    print(f"  ✓ Test:  {X_test.shape[0]:,} samples")

    # Free some memory
    del df, X, y
    import gc; gc.collect()

    # ── 5. Train model ────────────────────────────────────────────
    print("\n[5/6] Training Random Forest (25 trees, max_depth=15)...")
    start = time.time()
    model = RandomForestClassifier(
        n_estimators=25,
        random_state=42,
        n_jobs=-1,
        max_depth=15,
        min_samples_leaf=2,
        max_features='sqrt',
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"  ✓ Training completed in {train_time:.1f}s")

    # ── 6. Evaluate ───────────────────────────────────────────────
    print("\n[6/6] Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  ┌─────────────────────────────┐")
    print(f"  │  Test Accuracy: {acc*100:.2f}%     │")
    print(f"  └─────────────────────────────┘")

    # Print macro-averaged metrics
    report = classification_report(
        y_test, y_pred,
        output_dict=True,
        zero_division=0,
    )
    macro = report['macro avg']
    print(f"\n  Macro Avg — Precision: {macro['precision']:.4f}  "
          f"Recall: {macro['recall']:.4f}  "
          f"F1: {macro['f1-score']:.4f}")

    # ── 7. Export artifacts ───────────────────────────────────────
    print("\n[+] Exporting model artifacts to model/ ...")
    os.makedirs("model", exist_ok=True)

    joblib.dump(model, "model/disease_model.pkl")
    joblib.dump(le, "model/label_encoder.pkl")
    joblib.dump(list(symptom_columns), "model/symptom_columns.pkl")

    model_size_mb = os.path.getsize("model/disease_model.pkl") / (1024 * 1024)
    print(f"  ✓ model/disease_model.pkl     ({model_size_mb:.1f} MB)")
    print(f"  ✓ model/label_encoder.pkl")
    print(f"  ✓ model/symptom_columns.pkl")

    print("\n" + "=" * 60)
    print("  ✅  All done! Model is ready for the Flask webapp.")
    print("=" * 60)


if __name__ == "__main__":
    main()
