import streamlit as st

# Imports!

import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
from utils import precompute_results, compare_learning_curves

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


import warnings

warnings.filterwarnings("ignore")


@st.cache
def data_scaling(df: pd.DataFrame()):
    standard_scaler = StandardScaler()
    robust_scaler = RobustScaler()

    _df = df.copy()

    _df["Amount_scaled"] = standard_scaler.fit_transform(
        _df["Amount"].values.reshape(-1, 1)
    )

    _df["Time_scaled"] = standard_scaler.fit_transform(
        _df["Time"].values.reshape(-1, 1)
    )

    amount_scaled = _df["Amount_scaled"]
    time_scaled = _df["Time_scaled"]

    _df.drop(["Amount_scaled", "Time_scaled"], axis=1, inplace=True)
    _df.drop(["Amount", "Time"], axis=1, inplace=True)
    _df.insert(0, "amount_scaled", amount_scaled)
    _df.insert(1, "time_scaled", time_scaled)

    return _df


@st.cache
def df_corr(df: pd.DataFrame):
    return df.corr()


@st.cache
def create_xy(df: pd.DataFrame, label: str):
    return df.drop(label, axis=1), df[label]
