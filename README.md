<<<<<<< HEAD
# Fraud Detection using Bayesian Networks 

This project implements a fraud detection system using Bayesian Networks with support for latent variable learning.

## Features
- Bayesian Network-based fraud detection
- Multiple estimation methods (MLE, Bayes, EM)
- Automated bin optimization for continuous variables
- Latent variable learning and inference 
- Comprehensive visualization tools

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```

This will:
1. Generate synthetic fraud data
2. Train the model using different estimation methods
3. Create visualization plots
4. Save results to estimation_results.csv

## Project Structure

- `model.py`: Core Bayesian Network implementation
- `utils.py`: Helper functions for preprocessing and evaluation
- `data_generator.py`: Synthetic data generation
- `visualizations.py`: Plotting functions
- `main.py`: Main script to run experiments

## Results

The model outputs several visualizations:
- ROC curves
- Feature distributions
- Network structure
- Confusion matrices
- Feature importance plots
=======
# Fraud-detection-using-Bayesian-Network
>>>>>>> 516b6991c0ccb3d599015fb5767473d50ec41557
