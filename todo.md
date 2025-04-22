# Gradient Boosting Tree Classification Implementation

## Tasks

- [x] Retrieve and review sections 10.9-10.10 from Elements of Statistical Learning
- [ ] Understand gradient boosting algorithm in detail
- [ ] Create appropriate directory structure (similar to Project 1)
- [ ] Implement base decision tree classifier
- [ ] Implement gradient boosting classifier with fit-predict interface
- [ ] Develop test data and validation cases
- [ ] Test and validate implementation
- [ ] Document implementation and results

## Notes on Gradient Boosting Algorithm

### Key Components
1. Decision trees as base learners that partition feature space
2. Forward Stagewise Additive Modeling (FSAM) approach
3. Gradient boosting as numerical optimization technique
4. Fitting trees to pseudoresiduals (negative gradients)

### Algorithm Overview
The gradient boosting tree algorithm works by:
1. Starting with an initial prediction (typically a constant)
2. Computing the negative gradient (pseudoresiduals) of the loss function
3. Fitting a regression tree to these pseudoresiduals
4. Updating the model by adding the new tree with an appropriate step size
5. Repeating steps 2-4 for a specified number of iterations

### Mathematical Foundation
- Loss function: L(y, f(x))
- Gradient: -∂L(y, f(x))/∂f(x)
- Model update: f_m(x) = f_{m-1}(x) + ∑ γ_jm I(x ∈ R_jm)
