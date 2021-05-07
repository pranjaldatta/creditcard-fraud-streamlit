# Imports!

import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time

# Standard Scikit-Learn Imports
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    classification_report,
)

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

import streamlit as st


def precompute_results(models, x_train, y_train, x_test, y_test, gridcv_params):

    _precompute_results = {}
    _gridcv_optimized = {}

    count = 1
    for key, model in models.items():

        start_time = time.time()
        model.fit(x_train, y_train)
        end_time = time.time()

        score = cross_val_score(model, x_train, y_train, cv=5)

        _precompute_results[key] = {
            "model_name": model.__class__.__name__,
            "time": end_time - start_time,
            "cross_val_score": round(score.mean(), 2) * 100,
        }

        count += 1

    for key, model in models.items():
        _grid_search_model = GridSearchCV(model, gridcv_params[key])
        _grid_search_model.fit(x_train, y_train)
        _gridcv_optimized.update({key: _grid_search_model.best_estimator_})

    count = 1
    for key, optimized_model in _gridcv_optimized.items():
        score = cross_val_score(optimized_model, x_train, y_train, cv=5)
        _pred = optimized_model.predict(x_test)
        acc_score = accuracy_score(y_test, _pred)
        _gridcv_optimized.update(
            {
                key: {
                    "model": optimized_model,
                    "cross_val": round(score.mean(), 3) * 100,
                    "accuracy": round(acc_score.mean(), 2) * 100,
                }
            }
        )
        count += 1

    return _precompute_results, _gridcv_optimized


def compare_learning_curves(
    model,
    x,
    y,
    ylim=None,
    cv=None,
    n_jobs=1,
    f=None,
    ax=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):

    if ax is None:
        f, ax = plt.subplots(figsize=(10, 5), sharey=True)

    if ylim is not None:
        plt.ylim(ylim)

    train_sizes, train_scores, test_scores = learning_curve(
        model, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )

    mean_train_scores = np.mean(train_scores, axis=1)
    std_train_scores = np.std(train_scores, axis=1)

    mean_test_scores = np.mean(test_scores, axis=1)
    std_test_scores = np.std(test_scores, axis=1)

    ax.fill_between(
        train_sizes,
        mean_train_scores - std_train_scores,
        mean_train_scores + std_train_scores,
        alpha=0.1,
        color="#2492ff",
    )

    ax.fill_between(
        train_sizes,
        mean_test_scores - std_test_scores,
        mean_test_scores + std_test_scores,
        alpha=0.1,
        color="#00ff00",
    )

    ax.plot(train_sizes, mean_train_scores, "o-", color="#2492ff", label="Train Score")
    ax.plot(train_sizes, mean_test_scores, "o-", color="#00ff00", label="Test Score")

    ax.set_title(
        f"{model.__class__.__name__} Training Curve vs Validation Curve", fontsize=18
    )
    ax.set_xlabel("Training Sample Size")
    ax.set_ylabel("Scores")
    ax.grid(True)
    ax.legend(loc="best")

    return f, ax
