#!/usr/bin/env python3
"""
Simple ML WAF Training Script
Quick and easy model training
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import urllib.parse
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def extract_simple_features(df):
    """Extract simple features from dataset"""
    logging.info("Extracting features...")
    
    features = df.copy()
    
    # URL decode
    features['url_decoded'] = features['URL'].apply(lambda x: urllib.parse.unquote_plus(str(x)))
    features['body_decoded'] = features['content'].apply(lambda x: urllib.parse.unquote_plus(str(x)))
    
    # Basic features
    features['url_length'] = features['url_decoded'].str.len()
    features['body_length'] = features['body_decoded'].str.len()
    features['num_params'] = features['url_decoded'].str.count(r'\?') + features['url_decoded'].str.count(r'&')
    
    # Suspicious character counts
    suspicious_chars = ["'", '"', "<", ">", ";", "%"]
    for ch in suspicious_chars:
        features[f'count_{ch}'] = features['url_decoded'].str.count(ch) + features['body_decoded'].str.count(ch)
    
    # Attack keywords
    attack_patterns = {
        'has_sql': r'\b(select|union|insert|update|delete|drop|create)\b',
        'has_xss': r'<script|javascript:|alert\(',
        'has_command': r'\b(whoami|wget|curl|bash|python|perl)\b',
        'has_path_traversal': r'\.\./|\.\.\\',
        'has_comment': r'--|#|/\*'
    }
    
    combined_text = (features['url_decoded'] + " " + features['body_decoded']).str.lower()
    
    for name, pattern in attack_patterns.items():
        features[name] = combined_text.str.contains(pattern, case=False, regex=True).astype(int)
    
    return features

def main():
    """Main training function"""
    logging.info("Starting simple ML WAF training...")
    
    # Load data
    logging.info("Loading dataset...")
    df = pd.read_csv('csic_database.csv')
    
    # Check target column
    if 'label' in df.columns:
        target_col = 'label'
    elif 'class' in df.columns:
        target_col = 'class'
    elif 'classification' in df.columns:
        target_col = 'classification'
    else:
        logging.error("No target column found!")
        logging.info(f"Available columns: {list(df.columns)}")
        return
    
    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f"Target distribution: {df[target_col].value_counts()}")
    
    # Extract features
    df = extract_simple_features(df)
    
    # Select features
    feature_cols = ['url_length', 'body_length', 'num_params'] + \
                   [col for col in df.columns if col.startswith(('count_', 'has_'))]
    
    # Make sure all feature columns exist
    available_features = [col for col in feature_cols if col in df.columns]
    logging.info(f"Using features: {available_features}")
    
    X = df[available_features].fillna(0)
    
    # Target is already binary (0=normal, 1=attack)
    y = df[target_col].astype(int)
    
    logging.info(f"Target distribution: {y.value_counts()}")
    
    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Features: {list(X.columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    logging.info("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Evaluate Random Forest
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)
    
    # Handle case where model only predicts one class
    if rf_pred_proba.shape[1] == 1:
        rf_pred_proba = np.column_stack([1 - rf_pred_proba, rf_pred_proba])
    
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_pred_proba[:, 1])
    
    logging.info(f"Random Forest - Accuracy: {rf_accuracy:.4f}, AUC: {rf_auc:.4f}")
    
    # Train Logistic Regression
    logging.info("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # Evaluate Logistic Regression
    lr_pred = lr_model.predict(X_test)
    lr_pred_proba = lr_model.predict_proba(X_test)
    
    # Handle case where model only predicts one class
    if lr_pred_proba.shape[1] == 1:
        lr_pred_proba = np.column_stack([1 - lr_pred_proba, lr_pred_proba])
    
    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_pred_proba[:, 1])
    
    logging.info(f"Logistic Regression - Accuracy: {lr_accuracy:.4f}, AUC: {lr_auc:.4f}")
    
    # Save models
    logging.info("Saving models...")
    
    # Save Random Forest
    joblib.dump(rf_model, 'models/rf_model1.pkl')
    rf_config = {
        'model_type': 'RandomForest',
        'accuracy': float(rf_accuracy),
        'auc': float(rf_auc),
        'feature_columns': list(X.columns),
        'training_date': datetime.now().isoformat()
    }
    with open('models/model_config.json', 'w') as f:
        json.dump(rf_config, f, indent=2)
    
    # Save Logistic Regression
    joblib.dump(lr_model, 'models/better_model.pkl')
    lr_config = {
        'model_type': 'LogisticRegression',
        'accuracy': float(lr_accuracy),
        'auc': float(lr_auc),
        'feature_columns': list(X.columns),
        'training_date': datetime.now().isoformat()
    }
    with open('models/better_config.json', 'w') as f:
        json.dump(lr_config, f, indent=2)
    
    # Save feature names for later use
    feature_info = {
        'feature_columns': list(X.columns),
        'target_column': target_col
    }
    with open('models/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    logging.info("Training completed!")
    logging.info(f"Random Forest saved: models/rf_model1.pkl")
    logging.info(f"Logistic Regression saved: models/better_model.pkl")
    logging.info(f"Configs saved: models/model_config.json, models/better_config.json")

if __name__ == "__main__":
    main()
