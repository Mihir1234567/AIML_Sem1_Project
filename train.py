import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_save():
    start_time = time.time()
    print("🚀 Starting model training and artifact export...")
    
    # 1. Dataset Path
    possible_paths = [
        os.path.join("Notebook", "kickstarter_data.csv"),
        "kickstarter_data.csv"
    ]
    data_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if not data_path:
        raise FileNotFoundError("kickstarter_data.csv not found in workspace.")
    
    print(f"📥 Loading dataset from {data_path}...")
    raw_df = pd.read_csv(data_path, encoding='latin-1', low_memory=False)
    raw_df.columns = [col.strip() for col in raw_df.columns]
    
    # 2. Filter Finalized Campaign Outcomes (Drop Live, Suspended, Undefined)
    valid_states = ['successful', 'failed']
    df = raw_df[raw_df['state'].isin(valid_states)].copy()
    
    # Target
    df['target'] = (df['state'] == 'successful').astype(int)
    
    # Currency Normalization
    fx_rates = {
        'USD': 1.0, 'GBP': 1.53, 'EUR': 1.10, 'CAD': 0.80, 'AUD': 0.76,
        'SEK': 0.12, 'NZD': 0.71, 'DKK': 0.15, 'NOK': 0.12, 'CHF': 1.00,
        'SGD': 0.74, 'HKD': 0.13, 'MXN': 0.05
    }
    df['goal_numeric'] = pd.to_numeric(df['goal'], errors='coerce').fillna(1000.0)
    df['fx_rate'] = df['currency'].map(fx_rates).fillna(1.0)
    df['usd_goal_real'] = df['goal_numeric'] * df['fx_rate']
    
    # Timestamps & Duration
    df['launched_dt'] = pd.to_datetime(df['launched'], errors='coerce')
    df['deadline_dt'] = pd.to_datetime(df['deadline'], errors='coerce')
    df = df.dropna(subset=['launched_dt', 'deadline_dt']).copy()
    df['duration_days'] = (df['deadline_dt'] - df['launched_dt']).dt.total_seconds() / 86400.0
    df = df[(df['duration_days'] >= 1) & (df['duration_days'] <= 90)].copy()
    
    # Feature Extraction
    features_df = pd.DataFrame(index=df.index)
    features_df['target'] = df['target']
    features_df['main_category'] = df['main_category'].fillna('Unknown')
    features_df['country'] = df['country'].fillna('Unknown')
    features_df['log_usd_goal'] = np.log1p(df['usd_goal_real'])
    features_df['duration_days'] = df['duration_days']
    features_df['launch_month'] = df['launched_dt'].dt.month
    features_df['launch_day_of_week'] = df['launched_dt'].dt.dayofweek
    features_df['is_weekend'] = features_df['launch_day_of_week'].isin([5, 6]).astype(int)
    
    # Text Heuristics
    clean_names = df['name'].fillna('').astype(str)
    features_df['name_char_len'] = clean_names.apply(len)
    features_df['name_word_count'] = clean_names.apply(lambda s: len(s.split()))
    features_df['has_exclamation'] = clean_names.apply(lambda s: int('!' in s))
    features_df['has_question'] = clean_names.apply(lambda s: int('?' in s))
    features_df['is_all_caps'] = clean_names.apply(lambda s: int(s.isupper() and len(s) > 3))
    
    # Unique categories and countries for frontend dropdowns
    categories = sorted(features_df['main_category'].unique().tolist())
    countries = sorted(features_df['country'].unique().tolist())
    
    # Subsets
    numeric_features = [
        'log_usd_goal', 'duration_days', 'launch_month', 'launch_day_of_week',
        'is_weekend', 'name_char_len', 'name_word_count', 'has_exclamation',
        'has_question', 'is_all_caps'
    ]
    categorical_features = ['main_category', 'country']
    
    X = features_df[numeric_features + categorical_features]
    y = features_df['target']
    
    # Sample for training
    sample_size = min(60000, len(features_df))
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=sample_size, random_state=42, stratify=y
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.20, random_state=42, stratify=y_sample
    )
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])
    
    # Classifier: HistGradientBoosting offers high ROC-AUC and fast execution
    model = HistGradientBoostingClassifier(
        max_iter=120, max_depth=6, random_state=42
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    print("⏳ Fitting pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Evaluation on holdout test set
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_prob))
    }
    print(f"📊 Test Metrics: ROC-AUC={metrics['roc_auc']:.4f}, Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    # Save artifacts
    artifact_dir = "model_artifacts"
    os.makedirs(artifact_dir, exist_ok=True)
    pipeline_file = os.path.join(artifact_dir, "kickstarter_pipeline.joblib")
    joblib.dump(pipeline, pipeline_file)
    print(f"💾 Model pipeline saved to {pipeline_file}")
    
    metadata = {
        'model_name': 'HistGradientBoostingClassifier',
        'metrics': metrics,
        'categories': categories,
        'countries': countries,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'trained_at': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    metadata_file = os.path.join(artifact_dir, "model_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Model metadata saved to {metadata_file}")
    
    elapsed = time.time() - start_time
    print(f"✅ Training & export completed in {elapsed:.2f} seconds!")

if __name__ == '__main__':
    train_and_save()

