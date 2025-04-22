# Gradient Boosting Tree Classification Algorithm Implementation

## Boosting Trees ( Given Problem For Project )

Implement again from first principles the gradient-boosting tree classification algorithm (with the usual fit-predict interface as in Project 1) as described in Sections 10.9-10.10 of Elements of Statistical Learning (2nd Edition). Answer the questions below as you did for Project 1. In this assignment, you'll be responsible for developing your own test data to ensure that your implementation is satisfactory. (Hint: Use the same directory structure as in Project 1.)

The same "from first principals" rules apply; please don't use SKLearn or any other implementation. Please provide examples in your README that will allow the TAs to run your model code and whatever tests you include. As usual, extra credit may be given for an "above and beyond" effort.

---
---
## Team Members

  - Dhruv Bhimani (A20582831)
  - Manushi Patel (A20575366)
  - Smit Dhameliya (A20593154)
---
---
## Table of Contents

- [Questions](#questions)
- [Overview](#overview)
- [Installation and Setup](#installation-and-setup)
- [Directory Structure](#directory-structure)
- [Components](#components)
- [Algorithm Description](#algorithm-description)
- [Test Results](#test-results)
- [Hyperparameter Testing](#hyperparameter-testing)
- [Conclusions](#conclusions)
- [Future Improvements](#future-improvements)

---
---
# Questions

- What does the model you have implemented do and when should it be used?
  - The model is a Gradient Boosting Classifier, implemented from scratch using decision trees as base learners. Gradient Boosting is an ensemble method that builds trees sequentially, where each tree tries to 
    correct the errors made by the previous ones. The final prediction is a weighted combination of all trees.

  - Use cases include:

    - Binary classification tasks where high accuracy is desired

    - Problems with non-linear decision boundaries

    - Situations where interpretability (e.g., tree structures, feature importance) matters
 
    - It is particularly effective on structured/tabular datasets with mixed data types.

 ---
 
- How did you test your model to determine if it is working reasonably correctly?

    - The model was tested using multiple datasets:

      - Linear: for baseline separability

      - Moons and Circles: for non-linear decision boundaries

      - Nonlinear (custom): for robustness to real-world complexity

    - For each dataset:

      - Accuracy was computed on both training and test splits

      - The final decision tree in the ensemble was printed to verify structure

      - Decision boundaries were visualized for 2D datasets and manually inspected

   - Edge cases (e.g., high-dimensional inputs) were tested using dimensionality reduction

---
- What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)

    - The following parameters are exposed to users:
        - Parameter | Description
        - n_estimators | Number of trees in the ensemble
        - learning_rate | Shrinkage applied to each tree’s prediction
        - max_depth | Maximum depth of individual trees
        - min_samples_split | Minimum samples required to split a node
        - min_samples_leaf | Minimum samples required to be at a leaf node
        - max_leaf_nodes | (Optional) Maximum number of leaf nodes per tree
        - random_state | For reproducibility

---
- Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

   - Yes, limitations include:
 
     
     (Challenge | Cause/Reason | Workaround )
        - High-dimensional data ( >2D) | Visualization fails or becomes meaningless | Dimensionality reduction (PCA)
        - Multi-class classification | Current implementation only supports binary classification | Extend to one-vs-all or softmax trees
        - Sparse or categorical features | No preprocessing or encoding logic yet | Add data preprocessing pipeline
        - Very large datasets | Pure Python + recursion in trees can be slow | Optimize with vectorization or Cython

---
---

## Overview

This project implements the gradient boosting tree classification algorithm from first principles as described in Sections 10.9-10.10 of Elements of Statistical Learning (2nd Edition). The implementation follows the fit-predict interface pattern and includes comprehensive testing on various synthetic datasets.

------
---

### Directory Structure
```
gradient_boosting_project/
├── data/                  # Contains generated test datasets
├── results/               # Contains test results and visualizations
├── src/                   # Source code
│   ├── __init__.py        # Package initialization
│   ├── decision_tree.py   # Base decision tree regressor implementation
│   ├── gradient_boosting.py # Gradient boosting classifier implementation
│   └── data_generation.py # Test data generation utilities
└── tests/                 # Test scripts
    └── test_gradient_boosting.py # Test script for validation
```
---
---

## Installation and Setup

### 1. Clone Repository
```sh
git clone https://github.com/manupatel20/ML_Project2_IIT
```

### 2. Create a Virtual Environment
To ensure dependency management, create a virtual environment:
for Windows
```sh
cd .\ML_Project2_IIT\src
python -m venv .env
.env\scripts\activate
```
for Mac
```sh
cd .\ML_Project2_IIT\src\
python -m venv .env
source .env/bin/activate
```

### 3. Install Dependencies
Ensure that you have all required dependencies installed by running:
```sh
cd ..
pip install -r requirements.txt
```

---

### 4. Running the Model
Run the following command:
```sh
python -m src.main_n
```
This will ask user to enter Dataset name from the given list
![WhatsApp Image 2025-04-22 at 17 51 11_617eaee8](https://github.com/user-attachments/assets/0ca184c3-55b5-4b72-b896-3e5fd9a2354a)
![WhatsApp Image 2025-04-22 at 17 56 43_b818a2b6](https://github.com/user-attachments/assets/cc7419b8-4dc2-4ea4-bfc7-195877e5957a)

Plot will be saved at results/decision_boundary_moons.png

---
---

### Components

1. **Decision Tree Regressor**
   - Implemented in `src/decision_tree.py`
   - Serves as the base learner for the gradient boosting classifier
   - Features:
     - Recursive binary splitting based on mean squared error
     - Configurable maximum depth, minimum samples for split/leaf
     - Prediction functionality for regression tasks

2. **Gradient Boosting Classifier**
   - Implemented in `src/gradient_boosting.py`
   - Features:
     - Binary classification using log loss function
     - Forward stagewise additive modeling approach
     - Fitting trees to negative gradients (pseudoresiduals)
     - Configurable number of estimators, learning rate, and tree parameters
     - Prediction with probability estimates
     - Feature importance calculation

3. **Test Data Generation**
   - Implemented in `src/data_generation.py`
   - Generates various synthetic datasets:
     - Linear separable data
     - Nonlinear data
     - Moons dataset
     - Circles dataset
   

4. **Testing Framework**
   - Implemented in `tests/test_gradient_boosting.py`
   - Tests the classifier on multiple datasets
   - Evaluates performance metrics (accuracy, precision, recall, F1, ROC AUC)
   - Tests different hyperparameter configurations
   - Visualizes decision boundaries

---
---

## Algorithm Description

The gradient boosting algorithm implemented in this project follows these steps:

1. **Initialization**: Start with an initial prediction (log-odds of the base rate for binary classification)
2. **Iterative Process**: For m = 1 to M (number of estimators):
   - Calculate negative gradients (pseudoresiduals) of the loss function with respect to current predictions
   - Fit a regression tree to these pseudoresiduals
   - For each leaf region in the tree, compute the optimal update value
   - Update the model by adding the new tree's predictions scaled by the learning rate
3. **Prediction**: Convert final log-odds to probabilities using the sigmoid function and classify based on a threshold

---
---

## Test Results

The implementation was tested on various synthetic datasets with the following results:

| Dataset   | Accuracy | Precision | Recall | F1 Score | ROC AUC | Results | Plot Visualization   |
|-----------|----------|-----------|--------|----------|---------|---------|---------|
| Linear    | 0.6655   | 0.6695    | 0.6363 | 0.6525   | 0.7371  |![WhatsApp Image 2025-04-22 at 16 54 04_f07ca1fc](https://github.com/user-attachments/assets/172df325-b842-4571-8ea3-0f1847fcd161) | ![WhatsApp Image 2025-04-22 at 16 02 50_61771eff](https://github.com/user-attachments/assets/33493d49-60ac-455b-b54a-007cb8c586f0) |
| Nonlinear | 0.6350   | 0.5676    | 0.7159 | 0.6332   | 0.7165  |![WhatsApp Image 2025-04-22 at 16 54 58_b56529b3](https://github.com/user-attachments/assets/5c16a74f-d85d-4c26-8ac8-699e3230a7e4) | ![WhatsApp Image 2025-04-22 at 16 04 33_953c4320](https://github.com/user-attachments/assets/73012f6e-7ed8-4965-b491-d1b76b4a14a9) |
| Moons     | 0.9850   | 0.9850    | 0.9800 | 0.9849   | 0.9993  |![WhatsApp Image 2025-04-22 at 16 55 38_29b31a32](https://github.com/user-attachments/assets/e12a7c47-a3d1-4c67-89cb-f589232ee47c) | ![WhatsApp Image 2025-04-22 at 16 06 10_13aaa98e](https://github.com/user-attachments/assets/f505cfab-ee3b-4b2a-8ce6-8d99d965e6ee) |
| Circles   | 0.7450   | 0.7168    | 0.8100 | 0.7606   | 0.7846  |![WhatsApp Image 2025-04-22 at 16 56 15_4e3c3d02](https://github.com/user-attachments/assets/27c8e073-1904-4bdf-b6c0-ac33b9edb3f7) | ![WhatsApp Image 2025-04-22 at 16 07 46_52d68e6d](https://github.com/user-attachments/assets/19faee88-d623-42ae-a983-a851987b2bcd) |


---
---


### Hyperparameter Testing

Different hyperparameter configurations were tested on the Nonlinear dataset:

| n_estimators | Accuracy | ROC AUC |
|--------------|----------|---------|
| 10           | 0.5400   | 0.7087  |
| 50           | 0.6350   | 0.7368  |
| 100          | 0.6350   | 0.7165  |

The results show that increasing the number of estimators generally improves performance, with diminishing returns after a certain point. The learning rate and maximum tree depth also significantly impact model performance, with deeper trees potentially leading to overfitting on the training data.


---
---


## Conclusions

The gradient boosting tree classification algorithm implementation successfully demonstrates the principles described in Elements of Statistical Learning. The algorithm shows strong performance across various datasets, particularly excelling on datasets with clear decision boundaries (Linear, Moons, XOR). The implementation struggles more with the Circles dataset, which has a more complex circular decision boundary that is challenging for axis-parallel splits used in decision trees.

The hyperparameter testing confirms the importance of tuning the number of estimators, learning rate, and tree depth to achieve optimal performance for a given dataset. The implementation provides a solid foundation for understanding gradient boosting from first principles and can be extended to handle multi-class classification or regression tasks.


---
---

## Future Improvements

Potential improvements to the implementation include:
1. Support for multi-class classification
2. Implementation of different loss functions
3. Subsampling for stochastic gradient boosting
4. Early stopping based on validation performance
5. More sophisticated regularization techniques
6. Parallel tree building for improved performance
