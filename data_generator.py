import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_history_features(n_samples, is_fraud):
    """Generate historical fraud-related features"""
    if is_fraud:
        # Higher risk patterns for fraudulent transactions
        device_fraud = np.random.poisson(2, n_samples)  # More frauds per device
        ip_fraud = np.random.poisson(3, n_samples)      # More frauds per IP
        card_fraud = np.random.poisson(1.5, n_samples)  # More frauds per card
    else:
        # Lower risk patterns for legitimate transactions
        device_fraud = np.random.poisson(0.1, n_samples)
        ip_fraud = np.random.poisson(0.2, n_samples)
        card_fraud = np.random.poisson(0.1, n_samples)
    
    return device_fraud, ip_fraud, card_fraud

def generate_velocity_features(n_samples, is_fraud):
    """Generate velocity-based features"""
    if is_fraud:
        # Higher velocity patterns for fraudulent transactions
        time_since_last = np.random.exponential(1, n_samples)  # Shorter intervals
        transaction_velocity = np.random.poisson(8, n_samples)  # More transactions
        amount_velocity = np.random.lognormal(4, 1, n_samples)  # Higher amounts
    else:
        # Lower velocity patterns for legitimate transactions
        time_since_last = np.random.exponential(24, n_samples)  # Longer intervals
        transaction_velocity = np.random.poisson(2, n_samples)   # Fewer transactions
        amount_velocity = np.random.lognormal(3, 0.5, n_samples) # Lower amounts
    
    return time_since_last, transaction_velocity, amount_velocity

def generate_data(n_samples=20000, fraud_rate=0.13):
    """
    Generate synthetic transaction data with realistic fraud patterns
    """
    n_fraud = int(n_samples * fraud_rate)
    n_legitimate = n_samples - n_fraud
    
    # Generate legitimate transactions
    device_fraud_leg, ip_fraud_leg, card_fraud_leg = generate_history_features(n_legitimate, False)
    time_since_last_leg, trans_vel_leg, amount_vel_leg = generate_velocity_features(n_legitimate, False)
    
    legitimate_data = {
        'amount': np.random.lognormal(4, 1, n_legitimate),  # Most transactions between $20-$200
        'time_of_day': np.random.normal(14, 4, n_legitimate),  # Most activity during business hours
        'day_of_week': np.random.randint(0, 7, n_legitimate),
        'customer_age': np.random.normal(40, 15, n_legitimate),
        'account_age': np.random.normal(1000, 500, n_legitimate),  # Days
        'prev_transactions': np.random.poisson(30, n_legitimate),
        'device_fraud_count': device_fraud_leg,
        'ip_fraud_count': ip_fraud_leg,
        'card_fraud_count': card_fraud_leg,
        'time_since_last': time_since_last_leg,
        'transaction_velocity': trans_vel_leg,
        'amount_velocity': amount_vel_leg,
        'fraud': np.zeros(n_legitimate)
    }
    
    # Generate fraudulent transactions
    device_fraud_fr, ip_fraud_fr, card_fraud_fr = generate_history_features(n_fraud, True)
    time_since_last_fr, trans_vel_fr, amount_vel_fr = generate_velocity_features(n_fraud, True)
    
    fraudulent_data = {
        'amount': np.random.lognormal(6, 2, n_fraud),  # Higher amounts
        'time_of_day': np.random.normal(3, 2, n_fraud),  # More night activity
        'day_of_week': np.random.randint(0, 7, n_fraud),
        'customer_age': np.random.normal(35, 12, n_fraud),  # Slightly younger
        'account_age': np.random.normal(100, 200, n_fraud),  # Newer accounts
        'prev_transactions': np.random.poisson(5, n_fraud),  # Fewer previous transactions
        'device_fraud_count': device_fraud_fr,
        'ip_fraud_count': ip_fraud_fr,
        'card_fraud_count': card_fraud_fr,
        'time_since_last': time_since_last_fr,
        'transaction_velocity': trans_vel_fr,
        'amount_velocity': amount_vel_fr,
        'fraud': np.ones(n_fraud)
    }
    
    # Combine and shuffle data
    data = {}
    for key in legitimate_data.keys():
        data[key] = np.concatenate([legitimate_data[key], fraudulent_data[key]])
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Clean up data
    df['time_of_day'] = df['time_of_day'].clip(0, 23)
    df['customer_age'] = df['customer_age'].clip(18, 100)
    df['account_age'] = df['account_age'].clip(1, None)
    df['prev_transactions'] = df['prev_transactions'].clip(0, None)
    df['time_since_last'] = df['time_since_last'].clip(0, None)
    df['transaction_velocity'] = df['transaction_velocity'].clip(0, None)
    df['amount_velocity'] = df['amount_velocity'].clip(0, None)
    df['device_fraud_count'] = df['device_fraud_count'].clip(0, None)
    df['ip_fraud_count'] = df['ip_fraud_count'].clip(0, None)
    df['card_fraud_count'] = df['card_fraud_count'].clip(0, None)
    
    # Shuffle rows
    return df.sample(frac=1).reset_index(drop=True)

if __name__ == "__main__":
    # Generate data
    data = generate_data()
    
    # Save to CSV
    data.to_csv('bank_transactions.csv', index=False)
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total transactions: {len(data)}")
    print(f"Fraud transactions: {data['fraud'].sum()}")
    print(f"Fraud rate: {(data['fraud'].sum() / len(data)) * 100:.2f}%")