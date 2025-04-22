

import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
import matplotlib.pyplot as plt
from .decision_tree import DecisionTreeRegressor


class GradientBoostingClassifier:

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_leaf_nodes: Optional[int] = None,
                 random_state: Optional[int] = None):
     
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        
        self.trees = []
        self.initial_prediction = None
        self.classes_ = None
        
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
       
        # Clip values to avoid overflow
        x = np.clip(x, -100, 100)
        return 1 / (1 + np.exp(-x))
    
    def _log_loss_gradient(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
     
        p = self._sigmoid(pred)
        # Return the gradient
        return y - p
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingClassifier':
       
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Store the classes
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("GradientBoostingClassifier only supports binary classification.")
        
        # Initialize with log-odds of the base rate
        pos_rate = np.mean(y)
        # Avoid division by zero or log(0)
        pos_rate = np.clip(pos_rate, 1e-15, 1 - 1e-15)
        self.initial_prediction = np.log(pos_rate / (1 - pos_rate))
        
        # Initialize predictions with the initial value
        F = np.full(y.shape, self.initial_prediction)
        
        # Boosting iterations
        for m in range(self.n_estimators):
            # Calculate the negative gradient (pseudoresiduals)
            residuals = self._log_loss_gradient(y, F)
            
            # Fit a regression tree to the residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_leaf_nodes=self.max_leaf_nodes
            )
            tree.fit(X, residuals)
            
            # Update the predictions
            update = tree.predict(X)
            F += self.learning_rate * update
            
            # Store the tree
            self.trees.append(tree)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
      
        if self.initial_prediction is None:
            raise ValueError("Classifier has not been fitted yet.")
        
        # Start with the initial prediction
        F = np.full(X.shape[0], self.initial_prediction)
        
        # Add the contributions from each tree
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        
        # Convert log-odds to probabilities
        proba_1 = self._sigmoid(F)
        proba_0 = 1 - proba_1
        
        # Return probabilities for both classes
        return np.column_stack((proba_0, proba_1))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
      
        if self.classes_ is None:
            raise ValueError("Classifier has not been fitted yet.")
        
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
       
        return np.mean(self.predict(X) == y)
    
    def staged_predict_proba(self, X: np.ndarray) -> List[np.ndarray]:
      
        if self.initial_prediction is None:
            raise ValueError("Classifier has not been fitted yet.")
        
        # Start with the initial prediction
        F = np.full(X.shape[0], self.initial_prediction)
        staged_probas = []
        
        # Add the contributions from each tree
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
            
            # Convert log-odds to probabilities
            proba_1 = self._sigmoid(F.copy())
            proba_0 = 1 - proba_1
            
            # Store probabilities for this stage
            staged_probas.append(np.column_stack((proba_0, proba_1)))
        
        return staged_probas
    
    def feature_importances(self, feature_names=None):
       
        if not self.trees:
            raise ValueError("Classifier has not been fitted yet.")
        
        # Count the number of times each feature is used for splitting
        feature_counts = {}
        
        def count_features(node):
            if not node['is_leaf']:
                feature_idx = node['feature_idx']
                feature_counts[feature_idx] = feature_counts.get(feature_idx, 0) + 1
                count_features(node['left'])
                count_features(node['right'])
        
        # Count features in all trees
        for tree in self.trees:
            count_features(tree.tree)
        
        # Normalize to get importances
        total = sum(feature_counts.values())
        importances = {k: v / total for k, v in feature_counts.items()} if total > 0 else {}
        
        # Map indices to names if provided
        if feature_names is not None:
            importances = {feature_names[idx]: value for idx, value in importances.items()}
        
        return importances
    
    def plot_feature_importances(self, feature_names=None, figsize=(10, 6)):
       
        importances = self.feature_importances(feature_names)
        
        # Sort importances
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        features, values = zip(*sorted_importances)
        
        # Plot
        plt.figure(figsize=figsize)
        plt.barh(range(len(features)), values, align='center')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.show()

