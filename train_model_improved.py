"""
Disease Prediction Model Training Script - IMPROVED VERSION
===========================================================
Tests multiple algorithms and saves the best performing model.
Algorithms tested:
- Random Forest
- Gradient Boosting
- XGBoost
- Extra Trees
- Logistic Regression
- Support Vector Machine
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Run: pip install xgboost")


def load_and_prepare_data():
    """Load and preprocess the dataset."""
    print("\n[1/7] Loading dataset...")
    start = time.time()
    df = pd.read_csv("unified_820_diseases_symptoms_dataset_ROWLEVEL_CLEANED.csv")
    print(f"  ✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns in {time.time()-start:.1f}s")

    print("\n[2/7] Cleaning columns...")
    df.columns = df.columns.str.strip()

    # Drop duplicate columns
    duplicate_cols = [c for c in df.columns if c.endswith('.1')]
    if duplicate_cols:
        df.drop(columns=duplicate_cols, inplace=True)
        print(f"  ✓ Dropped {len(duplicate_cols)} duplicate columns")

    target_col = 'diseases'
    symptom_columns = [c for c in df.columns if c != target_col]

    # Filter out rare diseases
    counts = df[target_col].value_counts()
    rare = counts[counts < 2].index.tolist()
    if rare:
        df = df[~df[target_col].isin(rare)]
        print(f"  ✓ Filtered out {len(rare)} diseases with < 2 samples")

    X = df[symptom_columns].astype(np.int8)
    y = df[target_col]

    print(f"  ✓ Features: {X.shape[1]} symptoms")
    print(f"  ✓ Target: {y.nunique()} unique diseases ({len(df):,} rows)")

    return X, y, symptom_columns


def encode_target(y):
    """Encode target labels."""
    print("\n[3/7] Encoding target labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"  ✓ Encoded {len(le.classes_)} disease classes")
    return y_encoded, le


def split_data(X, y_encoded):
    """Split data into train and test sets."""
    print("\n[4/7] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"  ✓ Train: {X_train.shape[0]:,} samples")
    print(f"  ✓ Test:  {X_test.shape[0]:,} samples")
    return X_train, X_test, y_train, y_test


def get_models():
    """Define all models to test."""
    models = {
        'Random Forest (100 trees)': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=20,
            min_samples_leaf=2,
            max_features='sqrt',
        ),
        'Random Forest (200 trees)': RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            max_depth=25,
            min_samples_leaf=1,
            max_features='sqrt',
        ),
        'Extra Trees (100 trees)': ExtraTreesClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=25,
            min_samples_leaf=1,
            max_features='sqrt',
        ),
        'Extra Trees (200 trees)': ExtraTreesClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            max_depth=30,
            min_samples_leaf=1,
            max_features='sqrt',
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=10,
            random_state=42,
        ),
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        models['XGBoost (200 trees)'] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
    
    return models


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate all models."""
    print("\n[5/7] Training and evaluating multiple algorithms...")
    print("=" * 80)
    
    models = get_models()
    results = []
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        start = time.time()
        
        # Train
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get classification report for macro metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        macro_f1 = report['macro avg']['f1-score']
        
        results.append({
            'name': name,
            'model': model,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'train_time': train_time
        })
        
        print(f"  ✓ Accuracy: {accuracy*100:.2f}% | Macro F1: {macro_f1:.4f} | Time: {train_time:.1f}s")
    
    return results


def select_best_model(results):
    """Select the best model based on accuracy."""
    print("\n[6/7] Model Comparison Results:")
    print("=" * 80)
    
    # Sort by accuracy
    results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Algorithm':<30} {'Accuracy':<12} {'Macro F1':<12} {'Time (s)':<10}")
    print("-" * 80)
    
    for i, r in enumerate(results_sorted, 1):
        print(f"{i:<6} {r['name']:<30} {r['accuracy']*100:>10.2f}% {r['macro_f1']:>10.4f}  {r['train_time']:>8.1f}")
    
    best = results_sorted[0]
    print("\n" + "=" * 80)
    print(f"🏆 BEST MODEL: {best['name']}")
    print(f"   Test Accuracy: {best['accuracy']*100:.2f}%")
    print(f"   Macro F1: {best['macro_f1']:.4f}")
    print("=" * 80)
    
    return best


def save_model(best, le, symptom_columns):
    """Save the best model and artifacts."""
    print("\n[7/7] Exporting best model artifacts to model/ ...")
    os.makedirs("model", exist_ok=True)
    
    joblib.dump(best['model'], "model/disease_model.pkl")
    joblib.dump(le, "model/label_encoder.pkl")
    joblib.dump(list(symptom_columns), "model/symptom_columns.pkl")
    
    # Save metadata
    metadata = {
        'algorithm': best['name'],
        'accuracy': best['accuracy'],
        'macro_f1': best['macro_f1'],
        'train_time': best['train_time']
    }
    joblib.dump(metadata, "model/model_metadata.pkl")
    
    model_size_mb = os.path.getsize("model/disease_model.pkl") / (1024 * 1024)
    print(f"  ✓ model/disease_model.pkl     ({model_size_mb:.1f} MB)")
    print(f"  ✓ model/label_encoder.pkl")
    print(f"  ✓ model/symptom_columns.pkl")
    print(f"  ✓ model/model_metadata.pkl")


def main():
    print("=" * 80)
    print("  Disease Prediction Model — IMPROVED Training Pipeline")
    print("=" * 80)
    
    # Load and prepare data
    X, y, symptom_columns = load_and_prepare_data()
    
    # Encode target
    y_encoded, le = encode_target(y)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y_encoded)
    
    # Train and evaluate all models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Select best model
    best = select_best_model(results)
    
    # Save best model
    save_model(best, le, symptom_columns)
    
    print("\n" + "=" * 80)
    print("  ✅  All done! Best model is ready for the Flask webapp.")
    print("=" * 80)


if __name__ == "__main__":
    main()
