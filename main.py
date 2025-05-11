from data_generator import generate_data
from model import FraudDetectionBN
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from visualizations import (
    plot_feature_distributions,
    plot_roc_curves,
    plot_network_structure,
    plot_feature_importance,
    plot_latent_variable_analysis
)

def plot_confusion_matrix(y_true, y_pred, method_name):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {method_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{method_name.lower()}.png')
    plt.close()
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)  # Same as recall
    
    print(f"\nDetailed Metrics for {method_name}:")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    print(f"Specificity: {specificity:.2%}")
    print(f"Sensitivity: {sensitivity:.2%}")

def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_data(n_samples=20000)
    
    # Plot feature distributions before any preprocessing
    print("Plotting feature distributions...")
    plot_feature_distributions(data)
    
    # Print dataset statistics
    n_fraud = data['fraud'].sum()
    print(f"\nDataset Statistics:")
    print(f"Total transactions: {len(data)}")
    print(f"Fraud transactions: {n_fraud}")
    print(f"Fraud rate: {(n_fraud / len(data)) * 100:.2f}%")
    
    # Split into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['fraud'])
    
    # Dictionary to store results and probabilities
    results = {}
    probabilities = {}
    
    # Test all combinations of methods and bin tuning
    methods = {
        'mle': {},  # No latent variables
        'bayes': {},  # No latent variables
        'em': {
            'risk_level': ['amount', 'transaction_velocity', 'device_fraud_count']
        }  # Latent variable with its parent nodes
    }
    
    bin_tuning_options = [False, True]
    
    # Initialize first model to visualize network structure
    initial_model = FraudDetectionBN()
    print("\nGenerating network structure visualization...")
    plot_network_structure(initial_model)
    
    for method, latent_vars in methods.items():
        for tune_bins in bin_tuning_options:
            model_name = f"{method}_{'tuned' if tune_bins else 'fixed'}_bins"
            print(f"\nTraining {model_name}...")
            
            model = FraudDetectionBN(latent_vars=latent_vars, tune_bins=tune_bins)
            
            try:
                # Train model
                model.fit(train_data, method=method)
                
                # Make predictions
                y_pred = model.predict(test_data.drop('fraud', axis=1))
                y_pred_proba = model.predict_proba(test_data.drop('fraud', axis=1))
                probabilities[model_name] = y_pred_proba
                
                # Evaluate model
                metrics = model.evaluate(test_data.drop('fraud', axis=1), test_data['fraud'])
                results[model_name] = metrics
                
                # Print results
                print(f"\n{model_name} Results:")
                for metric, value in metrics.items():
                    print(f"{metric}: {value:.4f}")
                
                # Plot confusion matrix
                plot_confusion_matrix(test_data['fraud'], y_pred, model_name)
                
                # Plot feature importance
                print(f"\nGenerating feature importance plot for {model_name}...")
                plot_feature_importance(model, test_data)
                
                # If using EM, analyze latent variables
                if method == 'em':
                    print("\nAnalyzing learned latent variable patterns...")
                    plot_latent_variable_analysis(model, test_data)
                    
                    # Additional latent variable analysis
                    latent_dist = model.get_latent_distribution(test_data)
                    risk_probs = latent_dist['risk_level'][:, 1]
                    corr = np.corrcoef(risk_probs, y_pred_proba)[0, 1]
                    print(f"\nCorrelation between risk level and fraud probability: {corr:.3f}")
                    
                    # Print optimal bin counts if tuning was enabled
                    if tune_bins:
                        print("\nOptimal bin counts:")
                        for feature, n_bins in model.optimal_bins.items():
                            print(f"{feature}: {n_bins} bins")
            
            except Exception as e:
                print(f"Error with {model_name}: {str(e)}")
    
    # Plot ROC curves comparison
    print("\nGenerating ROC curves comparison...")
    plot_roc_curves(test_data['fraud'], probabilities)
    
    # Compare all methods
    print("\nComparison of Methods:")
    metrics_df = pd.DataFrame(results).round(4)
    print(metrics_df)
    
    # Save results
    metrics_df.to_csv('estimation_results.csv')

if __name__ == "__main__":
    main()