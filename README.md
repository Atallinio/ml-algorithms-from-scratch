# Machine Learning Classifiers for Glass Identification

This repository contains implementations of four classic machine learning algorithms applied to the Glass Identification dataset from UCI. The project demonstrates how different classifiers perform on a real-world dataset and includes utilities for preprocessing, evaluation, and hyperparameter tuning.

## Table of Contents
- [Dataset](#dataset)
- [Algorithms Implemented](#algorithms-implemented)
- [Features](#features)
- [How to Use](#how-to-use)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## Dataset

The **Glass Identification dataset** contains 214 instances with 9 chemical features used to classify glass types:
- RI: Refractive Index
- Na2O: Sodium Oxide
- MgO: Magnesium Oxide
- Al2O3: Aluminum Oxide
- SiO2: Silicon Dioxide
- K2O: Potassium Oxide
- CaO: Calcium Oxide
- BaO: Barium Oxide
- Fe2O3: Iron Oxide

**Target classes**: 6 types of glass (labeled 1-7, with some numbers skipped)

## Algorithms Implemented

1. **Decision Tree**
   - Uses ID3 algorithm with information gain
   - Supports configurable max depth, min samples per node, and impurity thresholds
   - Includes feature bucketing for continuous values

2. **K-Nearest Neighbors (KNN)**
   - Euclidean distance metric
   - Majority vote classification
   - Configurable number of neighbors

3. **Naive Bayes**
   - Gaussian probability distributions
   - Handles continuous features
   - Calculates class probabilities

4. **Support Vector Machine (SVM)**
   - Linear SVM implementation
   - One-vs-All multiclass strategy
   - Hinge loss with L2 regularization

## Features

- **Data Preprocessing**:
  - Stratified train-test splitting (60-40)
  - Min-Max normalization
  - Random seed based on student ID for reproducibility

- **Model Evaluation**:
  - Accuracy measurement
  - Confusion matrices
  - Feature selection experiments
  - Grid search for hyperparameter tuning

- **Utilities**:
  - `stratified_shuffle_split()` for balanced splits
  - `minmax_scalar()` for normalization
  - `grid_search()` for hyperparameter optimization
  - `feature_selection()` to test feature importance

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ml-algorithms-from-scratch.git
   cd ml-algorithms-from-scratch

2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn

3. Run individual classifier scripts:
   ```bash
   python decisionTrees.py
   python knn.py
   python naiveBayes.py
   python svm.py

4. Or run the main script to see dataset info and run all files:
   ```bash
   python main.py


   
