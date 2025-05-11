import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data, test_size=0.2, random_state=42):
    """
    Preprocess the data including scaling and train-test split
    """
    # Create copy of data
    df = data.copy()
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # List of features to scale
    features_to_scale = ['amount', 'time_of_day', 'customer_age', 'account_age', 'prev_transactions']
    
    # Scale features
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    # Split features and target
    X = df.drop('fraud', axis=1)
    y = df['fraud']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def print_metrics(metrics, method_name):
    """
    Print evaluation metrics in a formatted way
    """
    print(f"\n{method_name} Results:")
    print("-" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1-Score:  {metrics['f1_score']:.2%}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.2%}")
    print("-" * 50)