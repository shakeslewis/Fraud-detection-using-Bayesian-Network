import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator, ExpectationMaximization
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin

class BinOptimizer(BaseEstimator, ClassifierMixin):
    """Helper class for optimizing number of bins per feature"""
    def __init__(self, feature_name, n_bins=5):
        self.feature_name = feature_name
        self.n_bins = n_bins
        self.discretizer = None
        
    def fit(self, X, y):
        self.discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode='ordinal',
            strategy='quantile'
        )
        X_disc = self.discretizer.fit_transform(X.reshape(-1, 1))
        return self
        
    def predict(self, X):
        if self.discretizer is None:
            raise ValueError("Must call fit before predict")
        return self.discretizer.transform(X.reshape(-1, 1))
        
    def score(self, X, y):
        # Use mutual information as scoring metric
        X_disc = self.predict(X)
        counts = np.zeros((int(X_disc.max()) + 1, 2))
        for i in range(len(X)):
            counts[int(X_disc[i]), int(y[i])] += 1
        
        # Calculate mutual information
        total = len(X)
        mi = 0
        for bin_idx in range(counts.shape[0]):
            for class_idx in range(2):
                if counts[bin_idx, class_idx] > 0:
                    p_bin = counts[bin_idx].sum() / total
                    p_class = counts[:, class_idx].sum() / total
                    p_joint = counts[bin_idx, class_idx] / total
                    mi += p_joint * np.log2(p_joint / (p_bin * p_class))
        return mi

class FraudDetectionBN:
    def __init__(self, n_bins=5, latent_vars=None, tune_bins=False):
        self.n_bins = n_bins
        self.tune_bins = tune_bins
        self.discretizers = {}
        self.latent_vars = latent_vars or {}
        self.optimal_bins = {}
        
        # Start with core edges
        edges = [
            # Core fraud indicators
            ('amount', 'fraud'),
            ('device_fraud_count', 'fraud'),
            ('card_fraud_count', 'fraud'),
            ('transaction_velocity', 'fraud'),
            ('time_since_last', 'fraud'),
            
            # Secondary fraud indicators
            ('account_age', 'fraud'),
            ('time_of_day', 'fraud'),
            
            # Key feature dependencies
            ('transaction_velocity', 'amount'),
            ('card_fraud_count', 'device_fraud_count'),
            ('account_age', 'transaction_velocity')
        ]
        
        # Add latent variable edges if specified
        if self.latent_vars:
            for var, parents in self.latent_vars.items():
                # Add edges from parents to latent variable
                edges.extend([(parent, var) for parent in parents])
                # Add edge from latent variable to fraud
                edges.append((var, 'fraud'))
                
        # Initialize the network with all edges
        self.model = DiscreteBayesianNetwork(edges)
        self.inference = None
        
        print("\nInitialized Bayesian Network structure:")
        print(f"Nodes: {self.model.nodes()}")
        print(f"Edges: {self.model.edges()}")
        if self.latent_vars:
            print(f"Latent variables: {self.latent_vars}")
            
    def _optimize_bins(self, data, feature):
        """Find optimal number of bins for a feature using grid search"""
        print(f"\nOptimizing bins for {feature}...")
        X = data[feature].values.reshape(-1, 1)
        y = data['fraud'].values
        
        # Define parameter grid based on feature characteristics
        if feature in ['device_fraud_count', 'card_fraud_count']:
            param_grid = {'n_bins': [2, 3, 4, 5]}  # Fewer bins for count features
        elif feature in ['time_of_day']:
            param_grid = {'n_bins': [4, 6, 8, 12, 24]}  # More bins for time
        else:
            param_grid = {'n_bins': [3, 4, 5, 6, 7, 8]}  # Standard range
            
        optimizer = BinOptimizer(feature)
        grid_search = GridSearchCV(
            optimizer,
            param_grid,
            cv=5,
            scoring='neg_log_loss',  # Changed from mutual_info_score to neg_log_loss
            n_jobs=-1
        )
        
        try:
            grid_search.fit(X, y)
            optimal_bins = grid_search.best_params_['n_bins']
            print(f"Optimal bins for {feature}: {optimal_bins}")
            return optimal_bins
        except Exception as e:
            print(f"Warning: Bin optimization failed for {feature}: {str(e)}")
            # Return default number of bins if optimization fails
            return self.n_bins
            
    def _discretize_data(self, data, fit=True):
        """Discretize continuous variables with optimized bin counts"""
        discretized_data = data.copy()
        
        # Base bin counts (will be optimized if tune_bins=True)
        feature_bins = {
            'amount': 5,
            'time_of_day': 4,
            'account_age': 4,
            'device_fraud_count': 3,
            'card_fraud_count': 3,
            'time_since_last': 4,
            'transaction_velocity': 4
        }
        
        # Optimize bins if requested and fitting
        if fit and self.tune_bins:
            print("\nOptimizing bin counts...")
            for feature in feature_bins.keys():
                if feature in data.columns:
                    self.optimal_bins[feature] = self._optimize_bins(data, feature)
        
        # Use optimal or default bins for discretization
        for feature, default_bins in feature_bins.items():
            if feature in data.columns:
                n_bins = self.optimal_bins.get(feature, default_bins)
                if fit:
                    discretizer = KBinsDiscretizer(
                        n_bins=n_bins,
                        encode='ordinal',
                        strategy='quantile'
                    )
                    discretized_data[feature] = discretizer.fit_transform(
                        data[feature].values.reshape(-1, 1)
                    ).astype(int)
                    self.discretizers[feature] = discretizer
                else:
                    if feature in self.discretizers:
                        discretized_data[feature] = self.discretizers[feature].transform(
                            data[feature].values.reshape(-1, 1)
                        ).astype(int)
        
        # Add latent variables with random initial values
        for var in self.latent_vars:
            if var not in discretized_data.columns:
                # Initialize with random binary values
                discretized_data[var] = np.random.randint(0, 2, size=len(discretized_data))
        
        return discretized_data

    def _initialize_latent_cpds(self):
        """Initialize CPDs for latent variables with correct dimensionality"""
        for var, parents in self.latent_vars.items():
            print(f"\nInitializing CPD for {var}")
            print(f"Parents: {parents}")
            
            # Calculate cardinality for parent variables
            evidence_card = []
            evidence_values = []
            for parent in parents:
                if parent in self.discretizers:
                    n_bins = int(self.discretizers[parent].n_bins_)
                    evidence_card.append(n_bins)
                    evidence_values.append(list(range(n_bins)))
                else:
                    evidence_card.append(2)  # Binary for non-discretized variables
                    evidence_values.append([0, 1])
            
            print(f"Evidence cardinality: {evidence_card}")
            
            # Calculate total number of parent configurations
            n_parent_configs = int(np.prod(evidence_card)) if evidence_card else 1
            print(f"Number of parent configurations: {n_parent_configs}")
            
            # Initialize CPD with Dirichlet prior
            alpha = np.ones(2)  # Symmetric Dirichlet prior
            values = np.zeros((2, n_parent_configs))
            
            # Generate probabilities for each parent configuration
            for i in range(n_parent_configs):
                values[:, i] = np.random.dirichlet(alpha)
            
            try:
                # Create and add CPD
                cpd = TabularCPD(
                    variable=var,
                    variable_card=2,  # Binary latent variables
                    values=values,
                    evidence=parents if parents else None,
                    evidence_card=evidence_card if parents else None
                )
                
                # Remove existing CPD if any
                if var in [cpd.variable for cpd in self.model.get_cpds()]:
                    self.model.remove_cpds(var)
                
                self.model.add_cpds(cpd)
                print(f"Successfully initialized CPD for {var}")
                print(f"CPD shape: {values.shape}")
                
            except Exception as e:
                print(f"Failed to initialize CPD for {var}")
                print(f"Error: {str(e)}")
                raise e

    def _expectation_step(self, X_disc):
        """E-step: Compute expected values of latent variables"""
        latent_probs = {}
        
        for var in self.latent_vars:
            print(f"\nE-step for {var}")
            probs = []
            for i, (_, row) in enumerate(X_disc.iterrows()):
                if i == 0:  # Print first iteration details
                    print("\nFirst row details:")
                    print(f"Row data: {dict(row)}")
                
                # Only include observed variables in evidence
                evidence = {col: int(val) for col, val in row.items() 
                          if col in self.model.nodes() 
                          and col != var 
                          and col != 'fraud'
                          and col not in self.latent_vars}
                
                if i == 0:  # Print first iteration evidence
                    print(f"Evidence variables: {evidence.keys()}")
                    print(f"Evidence values: {evidence}")
                
                try:
                    query_result = self.inference.query(variables=[var], evidence=evidence)
                    if i == 0:  # Print first iteration result
                        print(f"Query result for first row: {query_result.values}")
                    probs.append(query_result.values)
                except Exception as e:
                    print(f"\nError in inference:")
                    print(f"Row index: {i}")
                    print(f"Evidence: {evidence}")
                    print(f"Error: {str(e)}")
                    raise e
                    
            latent_probs[var] = np.array(probs)
            print(f"\nCompleted E-step for {var}")
            print(f"Latent probabilities shape: {latent_probs[var].shape}")
        
        return latent_probs
        
    def _maximization_step(self, X_disc, latent_probs):
        """M-step: Update parameters using expected latent values"""
        print("\nStarting M-step")
        for var, probs in latent_probs.items():
            print(f"\nM-step for {var}")
            print(f"Probability distribution statistics:")
            print(f"Mean: {probs.mean(axis=0)}")
            print(f"Std: {probs.std(axis=0)}")
            
            # Add expected values to data
            X_disc[var] = (probs[:, 1] >= 0.5).astype(int)
            print(f"Assigned binary values: {X_disc[var].value_counts()}")
            
            try:
                # Re-estimate CPDs with new expected values
                estimator = MaximumLikelihoodEstimator(self.model, X_disc)
                
                # Update latent variable CPD
                print(f"\nUpdating CPD for {var}")
                cpd = estimator.estimate_cpd(var)
                self.model.remove_cpds(var)
                self.model.add_cpds(cpd)
                print(f"Updated CPD shape: {cpd.values.shape}")
                
                # Update fraud node CPD
                print("\nUpdating fraud node CPD")
                fraud_cpd = estimator.estimate_cpd('fraud')
                self.model.remove_cpds('fraud')
                self.model.add_cpds(fraud_cpd)
                print(f"Updated fraud CPD shape: {fraud_cpd.values.shape}")
                
            except Exception as e:
                print(f"\nError in M-step:")
                print(f"Variable: {var}")
                print(f"Data shape: {X_disc.shape}")
                print(f"Columns: {X_disc.columns}")
                raise e
        
        print("\nCompleted M-step")
    
    def _compute_log_likelihood(self, X_disc):
        """Compute log-likelihood of the data"""
        log_lik = 0
        for _, row in X_disc.iterrows():
            evidence = {col: int(val) for col, val in row.items() 
                      if col in self.model.nodes()
                      and col != 'fraud'
                      and col not in self.latent_vars}
            
            try:
                query_result = self.inference.query(variables=['fraud'], evidence=evidence)
                prob = query_result.values[int(row['fraud'])]
                log_lik += np.log(prob + 1e-10)  # Add small constant to avoid log(0)
            except Exception as e:
                print(f"Error computing likelihood: {str(e)}")
                print(f"Evidence: {evidence}")
                raise e
                
        return log_lik

    def _validate_model(self):
        """Validate that all CPDs are properly initialized"""
        try:
            self.model.check_model()
            all_nodes = set(self.model.nodes())
            cpd_nodes = set(cpd.variable for cpd in self.model.get_cpds())
            
            if all_nodes != cpd_nodes:
                missing = all_nodes - cpd_nodes
                raise ValueError(f"Missing CPDs for nodes: {missing}")
                
            return True
        except Exception as e:
            print(f"Model validation failed: {str(e)}")
            return False
            
    def fit(self, data, method='mle', max_iter=100, tol=1e-4):
        """Fit the model using specified estimation method"""
        features_to_use = [node for node in self.model.nodes() 
                          if node != 'fraud' and node not in self.latent_vars]
        X = data[features_to_use]
        y = data['fraud']
        
        # Discretize continuous variables
        X_disc = self._discretize_data(X)
        X_disc['fraud'] = y
        
        try:
            if method == 'mle':
                estimator = MaximumLikelihoodEstimator(self.model, X_disc)
                for node in self.model.nodes():
                    if node not in self.latent_vars:
                        cpd = estimator.estimate_cpd(node)
                        self.model.add_cpds(cpd)
                    
            elif method == 'bayes':
                estimator = BayesianEstimator(self.model, X_disc)
                for node in self.model.nodes():
                    if node not in self.latent_vars:
                        cpd = estimator.estimate_cpd(node, prior_type='BDeu', equivalent_sample_size=5)
                        self.model.add_cpds(cpd)
            
            elif method == 'em':
                if not self.latent_vars:
                    raise ValueError("EM estimation requires latent variables to be specified")
                
                print("\nInitializing EM estimation...")
                print(f"Latent variables: {list(self.latent_vars.keys())}")
                print(f"Observable nodes: {[n for n in self.model.nodes() if n not in self.latent_vars]}")
                
                # Initialize observed variables with MLE
                mle = MaximumLikelihoodEstimator(self.model, X_disc)
                for node in self.model.nodes():
                    if node not in self.latent_vars:
                        print(f"Estimating CPD for observed node: {node}")
                        cpd = mle.estimate_cpd(node)
                        if node in [c.variable for c in self.model.get_cpds()]:
                            self.model.remove_cpds(node)
                        self.model.add_cpds(cpd)
                
                # Initialize latent variables
                print("\nInitializing latent variable CPDs...")
                self._initialize_latent_cpds()
                
                # Validate model before starting EM
                print("\nValidating model structure...")
                if not self._validate_model():
                    raise ValueError("Model validation failed before EM")
                
                # Initialize inference engine
                print("\nInitializing inference engine...")
                self.inference = VariableElimination(self.model)
                
                # Run EM iterations
                print("\nRunning EM algorithm...")
                prev_loglik = float('-inf')
                for iteration in range(max_iter):
                    print(f"\nIteration {iteration + 1}/{max_iter}")
                    
                    # E-step
                    latent_probs = self._expectation_step(X_disc)
                    
                    # M-step
                    self._maximization_step(X_disc, latent_probs)
                    
                    # Validate model after each iteration
                    if not self._validate_model():
                        raise ValueError(f"Model validation failed after iteration {iteration + 1}")
                    
                    # Check convergence
                    curr_loglik = self._compute_log_likelihood(X_disc)
                    improvement = curr_loglik - prev_loglik
                    print(f"Log-likelihood = {curr_loglik:.4f} (improvement: {improvement:.4f})")
                    
                    if abs(improvement) < tol:
                        print(f"Converged after {iteration + 1} iterations")
                        break
                        
                    prev_loglik = curr_loglik
                
                print("\nEM estimation completed successfully")
            
            else:
                raise ValueError("Method must be one of: 'mle', 'bayes', 'em'")
            
            # Final model validation
            print("\nValidating final model...")
            if not self._validate_model():
                raise ValueError("Final model validation failed")
                
            self.inference = VariableElimination(self.model)
            
        except Exception as e:
            raise Exception(f"Error in {method} estimation: {str(e)}")
    
    def predict_proba(self, X):
        """Predict fraud probabilities for new data"""
        if self.inference is None:
            raise Exception("Model must be fitted before prediction")
        
        features_to_use = [node for node in self.model.nodes() if node != 'fraud']
        X = X[features_to_use]
        X_disc = self._discretize_data(X, fit=False)
        probabilities = []
        
        for _, row in X_disc.iterrows():
            evidence = {col: int(val) for col, val in row.items() if col in self.model.nodes()}
            query_result = self.inference.query(variables=['fraud'], evidence=evidence)
            prob_fraud = query_result.values[1]
            probabilities.append(prob_fraud)
            
        return np.array(probabilities)

    def predict(self, X, threshold=0.5):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def evaluate(self, X, y, threshold=0.5):
        """Evaluate model performance"""
        y_pred = self.predict(X, threshold)
        y_pred_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        return metrics

    def get_latent_distribution(self, X):
        """Get the distribution of latent variables for given data"""
        if not self.latent_vars or self.inference is None:
            raise ValueError("Model must be fitted with latent variables")
        
        X_disc = self._discretize_data(X, fit=False)
        latent_distributions = {}
        
        for var in self.latent_vars:
            distributions = []
            for _, row in X_disc.iterrows():
                evidence = {col: int(val) for col, val in row.items() 
                          if col in self.model.nodes() and col != var}
                query_result = self.inference.query(variables=[var], evidence=evidence)
                distributions.append(query_result.values)
            latent_distributions[var] = np.array(distributions)
        
        return latent_distributions

    def get_latent_insights(self, X):
        """Get insights about the latent variable patterns"""
        if not self.latent_vars:
            raise ValueError("No latent variables in model")
            
        latent_dist = self.get_latent_distribution(X)
        insights = {}
        
        for var, dist in latent_dist.items():
            high_risk = dist[:, 1] >= 0.5
            insights[var] = {
                'mean_prob': dist[:, 1].mean(),
                'high_risk_rate': high_risk.mean(),
                'distribution': dist[:, 1]
            }
            
        return insights