import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from src.gradient_boosting import GradientBoostingClassifier
from src.data_generation import load_dataset
from typing import Optional, Dict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import os

def print_tree(self, node: Optional[Dict] = None, prefix: str = "", is_left: bool = True):
    if node is None:
        if self.tree is None:
            print("Tree has not been fitted yet.")
            return
        node = self.tree

    branch = "├── " if is_left else "└── "

    if node['is_leaf']:
        print(f"{prefix}{branch}Leaf: value = {node['value']:.4f}, samples = {node['samples']}")
    else:
        print(f"{prefix}{branch}Node: feature[{node['feature_idx']}] <= {node['threshold']:.4f} "
              f"(samples = {node['samples']}, info_gain = {node['info_gain']:.4f})")
        next_prefix = prefix + ("│   " if is_left else "    ")
        self.print_tree(node['left'], next_prefix, True)
        self.print_tree(node['right'], next_prefix, False)

def plot_decision_boundary(clf, X, y, dataset_name):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.title(f"Decision Boundary for {dataset_name} Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(f"results/decision_boundary_{dataset_name}.png")
    print(f"Plot saved: results/decision_boundary_{dataset_name}.png")

def main():
    print("Gradient Boosting Tree Classification from Scratch")
    print("------------------------------------------")

    dataset_name = input("Enter dataset name (linear, nonlinear, moons, circles): ").strip().lower()

    print("Loading dataset...")
    try:
        X_train, X_test, y_train, y_test = load_dataset(dataset_name)
        print(f"Loaded {dataset_name} dataset")
    except FileNotFoundError:
        print(f"Dataset '{dataset_name}' not found. Exiting.")
        return

    print(f"Dataset shape: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

    # If more than 2 features, slice for plotting
    if X_train.shape[1] > 2:
        print("Note: Dataset has more than 2 features. Plotting and training using first two features only.")
        X_train = X_train[:, :2]
        X_test = X_test[:, :2]

    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_accuracy = np.mean(y_pred_train == y_train)
    test_accuracy = np.mean(y_pred_test == y_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    probas = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probas)

    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {auc:.4f}")

    print("\nFinal Decision Tree in the Ensemble:")
    clf.trees[-1].print_tree()

    if X_train.shape[1] == 2:
        plot_decision_boundary(clf, X_test, y_test, dataset_name)

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    main()
