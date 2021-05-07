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

    df["Amount_scaled"] = robust_scaler.fit_transform(
        df["Amount"].values.reshape(-1, 1)
    )

    df["Time_scaled"] = robust_scaler.fit_transform(df["Time"].values.reshape(-1, 1))

    amount_scaled = df["Amount_scaled"]
    time_scaled = df["Time_scaled"]

    df.drop(["Amount_scaled", "Time_scaled"], axis=1, inplace=True)
    df.drop(["Amount", "Time"], axis=1, inplace=True)
    df.insert(0, "amount_scaled", amount_scaled)
    df.insert(1, "time_scaled", time_scaled)

    return df
