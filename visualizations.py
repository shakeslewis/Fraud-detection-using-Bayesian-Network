import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import networkx as nx

def plot_feature_distributions(data, features=None):
    """Plot distribution of features comparing fraud vs non-fraud transactions"""
    if features is None:
        features = [col for col in data.columns if col != 'fraud']
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 4*n_rows))
    
    for i, feature in enumerate(features, 1):
        plt.subplot(n_rows, n_cols, i)
        
        # Plot distributions
        sns.kdeplot(data=data[data['fraud']==0], x=feature, label='Legitimate', alpha=0.6)
        sns.kdeplot(data=data[data['fraud']==1], x=feature, label='Fraud', alpha=0.6)
        
        plt.title(f'{feature} Distribution')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()

def plot_roc_curves(y_true, y_prob_dict):
    """Plot ROC curves for different methods"""
    plt.figure(figsize=(8, 6))
    
    for method_name, y_prob in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{method_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    plt.close()

def plot_network_structure(model):
    """Visualize the Bayesian Network structure"""
    plt.figure(figsize=(12, 8))
    
    # Create networkx graph
    G = nx.DiGraph()
    G.add_edges_from(model.model.edges())
    
    # Set node colors
    node_colors = ['lightblue' if node != 'fraud' else 'lightcoral' for node in G.nodes()]
    
    # Set node positions using spring layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the network
    nx.draw(G, pos, 
           node_color=node_colors,
           node_size=2000,
           with_labels=True,
           node_shape='o',
           font_size=10,
           font_weight='bold',
           arrows=True,
           edge_color='gray',
           arrowsize=20)
    
    plt.title("Bayesian Network Structure")
    plt.savefig('network_structure.png')
    plt.close()

def plot_feature_importance(model, data):
    """Plot feature importance based on conditional probabilities"""
    features = [node for node in model.model.nodes() if node != 'fraud']
    importances = []
    
    # Calculate importance for each feature
    X_disc = model._discretize_data(data[features])
    X_disc['fraud'] = data['fraud']
    
    for feature in features:
        # Calculate P(fraud=1|feature) for each value of the feature
        feature_values = X_disc[feature].unique()
        max_effect = 0
        
        for value in feature_values:
            evidence = {feature: int(value)}
            query_result = model.inference.query(variables=['fraud'], evidence=evidence)
            prob_fraud = query_result.values[1]
            max_effect = max(max_effect, abs(prob_fraud - data['fraud'].mean()))
        
        importances.append(max_effect)
    
    # Create importance plot
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    plt.barh(y=range(len(features)), width=importance_df['Importance'])
    plt.yticks(range(len(features)), importance_df['Feature'])
    plt.xlabel('Feature Importance (Max Δ in P(fraud))')
    plt.title('Feature Importance in Fraud Detection')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_latent_variable_analysis(model, data, latent_var='risk_level'):
    """Analyze and plot the learned latent variable patterns"""
    if latent_var not in model.latent_vars:
        raise ValueError(f"Latent variable {latent_var} not found in model")
    
    # Get latent variable distributions
    latent_dist = model.get_latent_distribution(data)
    dist = latent_dist[latent_var]
    
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Distribution of latent variable states
    plt.subplot(1, 2, 1)
    sns.histplot(dist[:, 1], bins=30)
    plt.title(f'{latent_var} State Distribution')
    plt.xlabel('Probability of High Risk State')
    plt.ylabel('Count')
    
    # Plot 2: Relationship with fraud
    plt.subplot(1, 2, 2)
    fraud_probs = model.predict_proba(data)
    plt.scatter(dist[:, 1], fraud_probs, alpha=0.1)
    plt.xlabel(f'{latent_var} Probability')
    plt.ylabel('Fraud Probability')
    plt.title('Latent Variable vs Fraud Probability')
    
    plt.tight_layout()
    plt.savefig(f'latent_analysis_{latent_var}.png')
    plt.close()