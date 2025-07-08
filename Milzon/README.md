# 🚢 Titanic Survival Analyzer Homework
_Milzon | AIEngineering-8 | 2025_

---

## Introduction

This notebook is part of the AIEngineering-8 assignment.  
The goal is to select and complete 3 out of 5 Titanic machine learning experiments, exploring feature engineering, model comparison, and data-driven insight.

---

## 1. Import Libraries & Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# Any other libraries you need


+-----------------------+
| Load Titanic Dataset   |
+-----------+-----------+
            |
            v
+-----------------------+
| Feature Engineering    |
+-----------+-----------+
            |
            v
+---------------------------+
| Define Features & Target   |
+-----------+---------------+
            |
            v
+---------------------------+
| Prepare Data Function      |
+-----------+---------------+
            |
            v
+---------------------------+
| Prepare Full Dataset       |
+-----------+---------------+
            |
            v
+---------------------------+
| Split Train/Test Data      |
+-----------+---------------+
            |
            v
+---------------------------+
| Train Baseline Model       |
+-----------+---------------+
            |
            v
+---------------------------+
| Predict & Compute Accuracy |
+-----------+---------------+
            |
            v
+---------------------------+
| Loop: Remove One Feature   |
| Train & Evaluate Model     |
+-----------+---------------+
            |
            v
+---------------------------+
| Compare Accuracies         |
+-----------+---------------+
            |
            v
+---------------------------+
| Draw Conclusions           |
+---------------------------+

## Task 1: Feature Detective Workflow

This workflow outlines the steps taken to determine the impact of individual features on the Titanic survival prediction model.

1. **Load Titanic Dataset**: Import the dataset into a Pandas DataFrame for analysis.
2. **Feature Engineering**: Create new features such as FamilySize and convert categorical features into numeric format.
3. **Define Features & Target**: Specify the list of input features and the target variable (Survived).
4. **Prepare Data Function**: Develop a function to clean and prepare data subsets for modeling.
5. **Prepare Full Dataset**: Apply the preparation function to the full feature set.
6. **Split Train/Test Data**: Partition the dataset into training and testing sets.
7. **Train Baseline Model**: Build a logistic regression model using all features.
8. **Predict & Compute Accuracy**: Evaluate model accuracy on the test set.
9. **Loop: Remove One Feature**: Iteratively remove each feature, retrain the model, and measure accuracy.
10. **Compare Accuracies**: Analyze the effect of each feature removal on model performance.
11. **Draw Conclusions**: Identify the most and least important features based on accuracy impact.

This systematic approach helps uncover which features contribute most significantly to predicting survival on the Titanic.

---

## Summary & Reflections

Through this assignment, I deepened my understanding of feature importance, the power of proper age and title engineering, and the pros/cons of different ML models.  
Excited to keep exploring more on data-driven storytelling and model explainability!

---

*Feel free to fork this repo or connect with me for data discussions!*

---
