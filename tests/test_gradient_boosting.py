"""
Test script for validating the gradient boosting classifier implementation.

This script tests the gradient boosting classifier implementation on various
synthetic datasets and compares its performance with expected results.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys
import os

# Add the parent directory to the path to import the src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gradient_boosting import GradientBoostingClassifier
from src.data_generation import (
    generate_linear_binary_data,
    generate_nonlinear_binary_data,
    generate_moons_data,
    generate_circles_data,
    generate_xor_data
)


def evaluate_classifier(clf, X_train, X_test, y_train, y_test, dataset_name):
    """
    Evaluate a classifier on a dataset and print the results.
    
    Parameters:
    -----------
    clf : object
        The classifier to evaluate.
    X_train : np.ndarray
        Training feature matrix.
    X_test : np.ndarray
        Testing feature matrix.
    y_train : np.ndarray
        Training target vector.
    y_test : np.ndarray
        Testing target vector.
    dataset_name : str
        Name of the dataset.
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics.
    """
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Calculate additional metrics for test set
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    
    # Calculate ROC AUC if predict_proba is available
    test_roc_auc = None
    if hasattr(clf, 'predict_proba'):
        y_prob_test = clf.predict_proba(X_test)[:, 1]
        test_roc_auc = roc_auc_score(y_test, y_prob_test)
    
    # Print results
    print(f"\n=== {dataset_name} Dataset ===")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test precision: {test_precision:.4f}")
    print(f"Test recall: {test_recall:.4f}")
    print(f"Test F1 score: {test_f1:.4f}")
    if test_roc_auc is not None:
        print(f"Test ROC AUC: {test_roc_auc:.4f}")
    
    # Return metrics
    metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_roc_auc': test_roc_auc
    }
    
    return metrics


def plot_decision_boundary(clf, X, y, title, ax=None, h=0.02):
    """
    Plot the decision boundary of a classifier.
    
    Parameters:
    -----------
    clf : object
        The classifier with predict method.
    X : np.ndarray
        Feature matrix (must be 2D).
    y : np.ndarray
        Target vector.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, a new figure is created.
    h : float
        Mesh step size.
    """
    if X.shape[1] != 2:
        raise ValueError("X must have exactly 2 features for decision boundary plotting.")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict on the mesh grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Plot the data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    
    # Add legend and title
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    return ax


def test_on_all_datasets():
    """
    Test the gradient boosting classifier on all datasets.
    """
    # Define classifier parameters
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': 42
    }
    
    # Generate datasets
    datasets = {
        'Linear': generate_linear_binary_data(n_features=2),  # 2 features for visualization
        'Nonlinear': generate_nonlinear_binary_data(n_features=2, n_informative=2, n_redundant=0),
        'Moons': generate_moons_data(),
        'Circles': generate_circles_data(),
        'XOR': generate_xor_data()
    }
    
    # Initialize results dictionary
    results = {}
    
    # Create figure for decision boundaries
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Test on each dataset
    for i, (name, (X_train, X_test, y_train, y_test)) in enumerate(datasets.items()):
        print(f"\nTesting on {name} dataset...")
        
        # Initialize and evaluate classifier
        clf = GradientBoostingClassifier(**params)
        metrics = evaluate_classifier(clf, X_train, X_test, y_train, y_test, name)
        
        # Store results
        results[name] = metrics
        
        # Plot decision boundary if dataset has 2 features
        if X_train.shape[1] == 2 and i < len(axes):
            # Combine train and test for visualization
            X_combined = np.vstack((X_train, X_test))
            y_combined = np.hstack((y_train, y_test))
            
            plot_decision_boundary(
                clf, X_combined, y_combined, 
                f"{name}: Test Acc={metrics['test_accuracy']:.4f}", 
                ax=axes[i]
            )
    
    # Remove any unused axes
    for i in range(len(datasets), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('../results/decision_boundaries.png')
    plt.close()
    
    return results


def test_hyperparameters():
    """
    Test the effect of different hyperparameters on the gradient boosting classifier.
    """
    # Generate a dataset
    X_train, X_test, y_train, y_test = generate_nonlinear_binary_data()
    
    # Test different numbers of estimators
    n_estimators_list = [10, 50, 100, 200]
    n_estimators_results = []
    
    for n_estimators in n_estimators_list:
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        metrics = evaluate_classifier(clf, X_train, X_test, y_train, y_test, 
                                     f"Nonlinear (n_estimators={n_estimators})")
        n_estimators_results.append(metrics)
    
    # Test different learning rates
    learning_rate_list = [0.01, 0.1, 0.5, 1.0]
    learning_rate_results = []
    
    for learning_rate in learning_rate_list:
        clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=learning_rate,
            max_depth=3,
            random_state=42
        )
        metrics = evaluate_classifier(clf, X_train, X_test, y_train, y_test, 
                                     f"Nonlinear (learning_rate={learning_rate})")
        learning_rate_results.append(metrics)
    
    # Test different max depths
    max_depth_list = [1, 3, 5, 10]
    max_depth_results = []
    
    for max_depth in max_depth_list:
        clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=max_depth,
            random_state=42
        )
        metrics = evaluate_classifier(clf, X_train, X_test, y_train, y_test, 
                                     f"Nonlinear (max_depth={max_depth})")
        max_depth_results.append(metrics)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot n_estimators results
    axes[0].plot(n_estimators_list, [r['train_accuracy'] for r in n_estimators_results], 'o-', label='Train')
    axes[0].plot(n_estimators_list, [r['test_accuracy'] for r in n_estimators_results], 's-', label='Test')
    axes[0].set_xlabel('Number of Estimators')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Effect of Number of Estimators')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot learning_rate results
    axes[1].plot(learning_rate_list, [r['train_accuracy'] for r in learning_rate_results], 'o-', label='Train')
    axes[1].plot(learning_rate_list, [r['test_accuracy'] for r in learning_rate_results], 's-', label='Test')
    axes[1].set_xlabel('Learning Rate')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Effect of Learning Rate')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot max_depth results
    axes[2].plot(max_depth_list, [r['train_accuracy'] for r in max_depth_results], 'o-', label='Train')
    axes[2].plot(max_depth_list, [r['test_accuracy'] for r in max_depth_results], 's-', label='Test')
    axes[2].set_xlabel('Max Depth')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Effect of Max Depth')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('../results/hyperparameter_effects.png')
    plt.close()


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('../results', exist_ok=True)
    
    print("=== Testing Gradient Boosting Classifier ===")
    
    # Test on all datasets
    print("\n--- Testing on all datasets ---")
    results = test_on_all_datasets()
    
    # Test hyperparameters
    print("\n--- Testing hyperparameters ---")
    test_hyperparameters()
    
    print("\nAll tests completed successfully.")
