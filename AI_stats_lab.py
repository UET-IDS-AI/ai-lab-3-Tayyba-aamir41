"""
Linear & Logistic Regression Lab

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 everywhere required.
"""

import numpy as np

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# QUESTION 1 – Linear Regression Pipeline (Diabetes)
# =========================================================

def diabetes_linear_pipeline():
    # STEP 1: Load dataset
    data = load_diabetes()
    X = data.data
    y = data.target

    # STEP 2: Train-Test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3: Standardization (fit only on train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 4: Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # STEP 5: Predictions
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    # STEP 6: Top 3 important features
    coef_abs = np.abs(model.coef_)
    top_3_feature_indices = np.argsort(coef_abs)[-3:][::-1].tolist()

# Does the model overfit?
# We compare train R² and test R².
# If training score is much higher than test score,
# the model is overfitting that means it memorized training data.
# If both scores are close, the model is not overfitting
# and it generalizes well to new data.

# Why is feature scaling important?
# Different features can have different ranges.
# Large values can dominate small values in the model.
# Scaling makes all features comparable,
# which helps the model learn properly and perform better.

    return train_mse, test_mse, train_r2, test_r2, top_3_feature_indices

# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

def diabetes_cross_validation():

    data = load_diabetes()
    X = data.data
    y = data.target

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()

    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

    mean_r2 = scores.mean()
    std_r2 = scores.std()

    return mean_r2, std_r2

# What does standard deviation represent?
# Standard deviation shows how much the model performance
# changes across different folds.
# If the value is small, the model is stable.
# If the value is large, the model performance is inconsistent.

# How does CV reduce variance risk?
# Cross-validation trains and tests the model on different
# parts of the data multiple times.
# This prevents us from depending on just one split.
# It gives a more reliable and balanced estimate of performance.

# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================
def cancer_logistic_pipeline():

    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)

    # Predictions
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    precision = precision_score(y_test, test_pred)
    recall = recall_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)

    cm = confusion_matrix(y_test, test_pred)

    return train_accuracy, test_accuracy, precision, recall, f1
# What does a False Negative mean in medical context?
# A False Negative means the model predicts "No Disease"
# but the patient actually has the disease.
# In medical problems, this is very dangerous
# because the patient may not receive treatment.

# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================
def cancer_logistic_regularization():

    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    C_values = [0.01, 0.1, 1, 10, 100]

    results = {}

    for C in C_values:
        model = LogisticRegression(max_iter=5000, C=C)
        model.fit(X_train_scaled, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

        results[C] = (train_acc, test_acc)
    return results

# What happens when C is very small?
# Very small C means strong regularization.
# The model becomes simpler and may underfit.

# What happens when C is very large?
# Very large C means weak regularization.
# The model becomes more complex and may fit training data too closely.

# Which case leads to overfitting?
# Overfitting usually happens when C is very large,
# because the model tries to memorize the training data.

# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================
def cancer_cross_validation():

    data = load_breast_cancer()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=1, max_iter=5000)

    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

    mean_accuracy = scores.mean()
    std_accuracy = scores.std()
    return mean_accuracy, std_accuracy

# Why is cross-validation critical in medical diagnosis?
# In medical problems, wrong predictions can be very serious.
# Cross-validation checks the model on multiple data splits.
# This ensures the model is stable and reliable,
# not just performing well on one lucky split.
