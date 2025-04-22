
import numpy as np
from typing import Dict, Optional


class DecisionTreeRegressor:
 
    
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2, 
                 min_samples_leaf: int = 1, max_leaf_nodes: Optional[int] = None):
     
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.tree = None
        self.leaf_count = 0
        
    def _calculate_mse(self, y: np.ndarray) -> float:
    
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    
    def _calculate_best_split(self, X: np.ndarray, y: np.ndarray) -> Optional[Dict]:
       
        m, n = X.shape
        if m <= self.min_samples_split:
            return None
        
        # Calculate the initial error (before any split)
        parent_error = self._calculate_mse(y)
        best_error = float('inf')
        best_split = None
        
        for feature_idx in range(n):
            # Get unique values for the feature
            feature_values = np.unique(X[:, feature_idx])
            
            # Try each value as a threshold
            for threshold in feature_values:
                # Split the data
                left_indices = X[:, feature_idx] <= threshold
                right_indices = ~left_indices
                
                # Skip if either side doesn't have enough samples
                if (np.sum(left_indices) < self.min_samples_leaf or 
                    np.sum(right_indices) < self.min_samples_leaf):
                    continue
                
                # Calculate the weighted error after split
                left_error = self._calculate_mse(y[left_indices])
                right_error = self._calculate_mse(y[right_indices])
                n_left, n_right = np.sum(left_indices), np.sum(right_indices)
                split_error = (n_left * left_error + n_right * right_error) / m
                
                # Update the best split if this is better
                if split_error < best_error:
                    best_error = split_error
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices,
                        'error': best_error,
                        'info_gain': parent_error - best_error
                    }
        
        return best_split if best_split is not None and best_split['error'] < parent_error else None
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict:
       
        m, n = X.shape
        if (depth >= self.max_depth or 
            m <= self.min_samples_split or 
            (self.max_leaf_nodes is not None and self.leaf_count >= self.max_leaf_nodes)):
            return {'value': np.mean(y), 'is_leaf': True, 'samples': m}
        
        # Find the best split
        best_split = self._calculate_best_split(X, y)
        
        # If no good split is found, create a leaf node
        if best_split is None:
            return {'value': np.mean(y), 'is_leaf': True, 'samples': m}
        
        # Create an internal node and build subtrees
        left_indices = best_split['left_indices']
        right_indices = best_split['right_indices']
        
        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        # Increment leaf count if children are leaves
        if left_subtree['is_leaf']:
            self.leaf_count += 1
        if right_subtree['is_leaf']:
            self.leaf_count += 1
        
        # Return the internal node
        return {
            'feature_idx': best_split['feature_idx'],
            'threshold': best_split['threshold'],
            'left': left_subtree,
            'right': right_subtree,
            'is_leaf': False,
            'samples': m,
            'info_gain': best_split['info_gain']
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressor':
      
        self.leaf_count = 0
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x: np.ndarray, node: Dict) -> float:
     
        if node['is_leaf']:
            return node['value']
        
        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
       
        if self.tree is None:
            raise ValueError("Tree has not been fitted yet.")
        
        return np.array([self._predict_sample(x, self.tree) for x in X])
    
    def print_tree(self, node: Optional[Dict] = None, depth: int = 0) -> None:
        
        if node is None:
            if self.tree is None:
                print("Tree has not been fitted yet.")
                return
            node = self.tree
        
        indent = "  " * depth
        
        if node['is_leaf']:
            print(f"{indent}Leaf: value = {node['value']:.4f}, samples = {node['samples']}")
        else:
            print(f"{indent}Node: feature_idx = {node['feature_idx']}, threshold = {node['threshold']:.4f}, "
                  f"samples = {node['samples']}, info_gain = {node['info_gain']:.4f}")
            print(f"{indent}Left ->")
            self.print_tree(node['left'], depth + 1)
            print(f"{indent}Right ->")
            self.print_tree(node['right'], depth + 1)
