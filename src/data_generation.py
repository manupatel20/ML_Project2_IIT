import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split


def generate_linear_binary_data(
    n_samples: int = 10000, 
    n_features: int = 10, 
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
   
    np.random.seed(random_state)
    
    # Generate random feature matrix
    X = np.random.randn(n_samples, n_features)
    
    # Generate weights for linear combination
    weights = np.random.randn(n_features)
    
    # Generate target variable (binary classification)
    y_logits = X.dot(weights)
    y = (y_logits > 0).astype(int)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def generate_nonlinear_binary_data(
    n_samples: int = 1000, 
    n_features: int = 10, 
    n_informative: int = 5,
    n_redundant: int = 2,
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
   
    # Generate dataset using sklearn's make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        random_state=random_state
    )
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def generate_moons_data(
    n_samples: int = 1000, 
    noise: float = 0.1,
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  
    # Generate dataset using sklearn's make_moons
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def generate_circles_data(
    n_samples: int = 1000, 
    noise: float = 0.1,
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # Generate dataset using sklearn's make_circles
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def generate_high_dimensional_data_with_noise(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 2,
    n_redundant: int = 10,
    flip_y: float = 0.2,
    class_sep: float = 0.5,
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
   
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        flip_y=flip_y,
        class_sep=class_sep,
        random_state=random_state
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def save_dataset(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray,
    dataset_name: str,
    save_dir: str = 'data'
) -> None:
  
    np.save(f"{save_dir}/{dataset_name}_X_train.npy", X_train)
    np.save(f"{save_dir}/{dataset_name}_X_test.npy", X_test)
    np.save(f"{save_dir}/{dataset_name}_y_train.npy", y_train)
    np.save(f"{save_dir}/{dataset_name}_y_test.npy", y_test)


def load_dataset(
    dataset_name: str,
    load_dir: str = 'data'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
   
    X_train = np.load(f"{load_dir}/{dataset_name}_X_train.npy")
    X_test = np.load(f"{load_dir}/{dataset_name}_X_test.npy")
    y_train = np.load(f"{load_dir}/{dataset_name}_y_train.npy")
    y_test = np.load(f"{load_dir}/{dataset_name}_y_test.npy")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Generate and save all datasets
    import os
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate and save linear data
    X_train, X_test, y_train, y_test = generate_linear_binary_data()
    save_dataset(X_train, X_test, y_train, y_test, "linear", data_dir)
    print(f"Linear dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Generate and save nonlinear data
    X_train, X_test, y_train, y_test = generate_nonlinear_binary_data()
    save_dataset(X_train, X_test, y_train, y_test, "nonlinear", data_dir)
    print(f"Nonlinear dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Generate and save moons data
    X_train, X_test, y_train, y_test = generate_moons_data()
    save_dataset(X_train, X_test, y_train, y_test, "moons", data_dir)
    print(f"Moons dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Generate and save circles data
    X_train, X_test, y_train, y_test = generate_circles_data()
    save_dataset(X_train, X_test, y_train, y_test, "circles", data_dir)
    print(f"Circles dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

    # Generate and save high-dim data
    X_train, X_test, y_train, y_test = generate_high_dimensional_data_with_noise()
    save_dataset(X_train, X_test, y_train, y_test, "high-dim", data_dir)
    print(f"High dimesnsional dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
   
    print("All datasets generated and saved successfully.")
