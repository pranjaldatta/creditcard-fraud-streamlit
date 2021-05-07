# Imports!

import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
from utils import precompute_results, compare_learning_curves
from data_utils import data_scaling, df_corr, create_xy
from model_utils import *

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

import warnings

warnings.filterwarnings("ignore")
random.seed(42)

import streamlit as st

st.title("Credit Card Fraud Detection!")

df_raw = st.cache(pd.read_csv)("creditcard.csv")


# Some basic calcs
# Lets calculate the the class occurance as a % of the total
fraud_count = len(df_raw[df_raw["Class"] == 1])
valid_count = len(df_raw[df_raw["Class"] == 0])
total_count = len(df_raw)

# Scaling
df_scaled = data_scaling(df_raw)

# Other caches
# vanilla_corr = st.cache(df_scaled.corr())

X_Y_data_created = False

avail_models = {
    "DecisionTree": dtc_cache,
    "SVC": svc_cache,
    "KNN": knn_cache,
    "LogReg": logreg_cache,
}

gridcv_params = {
    "DecisionTree": {
        "criterion": ["gini", "entropy"],
        "max_depth": list(range(2, 4, 1)),
        "min_samples_leaf": list(range(5, 7, 1)),
    },
    "SVC": {"C": [0.5, 0.7, 0.9, 1], "kernel": ["rbf", "poly", "sigmoid", "linear"]},
    "KNN": {
        "n_neighbors": list(range(2, 5, 1)),
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    },
    "LogReg": {"penalty": ["l1", "l2"], "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
}

# Print shape and description of the data
choice = st.sidebar.selectbox(
    "Menu", ["Data Details", "Preprocessing", "Compare models"]
)
if choice == "Data Details":
    st.write("<h2>Dataset Metadata</h2>", unsafe_allow_html=True),
    st.write(
        "* No of _total datapoints_: {}".format(total_count), unsafe_allow_html=True
    )
    st.write(
        "* No of _fraud datapoints_: {}".format(fraud_count), unsafe_allow_html=True
    )
    st.write(
        "* No of _valid datapoints_: {}".format(valid_count), unsafe_allow_html=True
    )
    st.write(
        "* '%' of _fraud transactions_: {:.2f} %".format(
            (fraud_count / total_count) * 100, unsafe_allow_html=True
        )
    )
    st.write(
        "* '%' of _valid transactions_: {:.2f} %".format(
            (valid_count / total_count) * 100, unsafe_allow_html=True
        )
    )

    st.bar_chart(
        pd.Series(
            {"Valid": valid_count, "Fraud": fraud_count}, index=["Valid", "Fraud"]
        )
    )

    st.write("<h2>Dataframe Head</h2>", unsafe_allow_html=True)
    st.dataframe(df_scaled.iloc[np.random.randint(1, 100, size=(10,))])

    st.write("<h2>Correlation Heatmap</h2>", unsafe_allow_html=True)
    fig, axes = plt.subplots(figsize=(30, 15))
    sns.heatmap(df_corr(df_scaled), annot=True, fmt=".2g", ax=axes)
    st.write(fig, unsafe_allow_html=True)
    st.write("**As we can infer from the heatmap,**", unsafe_allow_html=True)
    st.write(
        "*Negative Correlations*: V17, V14, V12 and V10 are negatively correlated. Notice how the lower these values are, the more likely the end result will be a fraud transaction.",
        unsafe_allow_html=True,
    )
    st.write(
        "*Positive Correlations*: V2, V4, V11, and V19 are positively correlated. Notice how the higher these values are, the more likely the end result will be a fraud transaction.",
        unsafe_allow_html=True,
    )

elif choice == "Preprocessing":

    st.write("<h2>Data Preprocessing</h2>", unsafe_allow_html=True)

    scaled_fraud_count = df_scaled[df_scaled.Class == 1]
    scaled_valid_count = df_scaled[df_scaled.Class == 0]

    FRAUD_PER = 0.95

    num_fraud = int(FRAUD_PER * scaled_fraud_count.shape[0])

    fraud_idx = random.sample(range(0, scaled_fraud_count.shape[0]), num_fraud)
    valid_idx = random.sample(range(0, scaled_valid_count.shape[0]), num_fraud)

    fraud_subsample = scaled_fraud_count.iloc[fraud_idx]
    valid_subsample = scaled_valid_count.iloc[valid_idx]

    dataset_subsample = pd.concat([fraud_subsample, valid_subsample], axis=0)
    dataset_subsample = dataset_subsample.sample(frac=1, random_state=120)

    st.write(f"Scaled Fraud Subsample shape: {fraud_subsample.shape}")
    st.write(f"Scaled Valid Subsample Shape: {valid_subsample.shape}")

    fig, axes = plt.subplots(figsize=(10, 5))
    class_count = pd.value_counts(
        [fraud_subsample.Class, valid_subsample.Class], sort=True
    )
    class_count.plot(kind="bar")
    plt.title("Transaction Class Distribution")
    plt.xticks(range(2), ["Valid", "Fraud"])
    plt.xlabel("Classes")
    plt.ylabel("No of Instances")
    st.write(fig)

    st.write(
        "<h2>Correlation heatmap for preprocessed dataset</h2>", unsafe_allow_html=True
    )
    fig, axes = plt.subplots(figsize=(30, 15))
    sns.heatmap(
        df_corr(dataset_subsample),
        annot=True,
        cmap="coolwarm",
        fmt=".2g",
        ax=axes,
        vmin=-1,
        vmax=+1,
        center=0,
    )
    st.write(fig)

    st.write("**As we can infer from the heatmap,**", unsafe_allow_html=True)
    st.write(
        "*Negative Correlations*: V17, V14, V12 and V10 are negatively correlated. Notice how the lower these values are, the more likely the end result will be a fraud transaction.",
        unsafe_allow_html=True,
    )
    st.write(
        "*Positive Correlations*: V2, V4, V11, and V19 are positively correlated. Notice how the higher these values are, the more likely the end result will be a fraud transaction.",
        unsafe_allow_html=True,
    )

    X, Y = create_xy(dataset_subsample, "Class")

    X_Y_data_created = True

    choice = st.sidebar.radio("Anaytics Menu", ("General Preprocessing", "Clustering"))

    if choice == "Clustering":

        st.write("<h2> Clustering the data </h2>", unsafe_allow_html=True)

        start_time = time.time()
        X_tsne = tsne_cache(X)
        end_time = time.time()
        st.write(
            "* *TSNE reduction took*: {:.2f} s".format(end_time - start_time),
            unsafe_allow_html=True,
        )

        start_time = time.time()
        X_pca = pca_cache(X)
        end_time = time.time()
        st.write(
            "* *PCA reduction took*: {:.2f} s".format(end_time - start_time),
            unsafe_allow_html=True,
        )

        start_time = time.time()
        X_svd = truncated_svd_cache(X)
        end_time = time.time()
        st.write(
            "* *Truncated SVD reduction took*: {:.2f} s".format(end_time - start_time),
            unsafe_allow_html=True,
        )

        figure, axes = plt.subplots(1, 3, figsize=(30, 10))
        figure.suptitle(
            "Here we compare all the clsuters with each other visually!", fontsize=18
        )

        axes[0].grid(True)
        axes[0].set_title("TSNE Clustering", fontsize=18)
        axes[0].scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            c=(Y == 0),
            cmap="coolwarm_r",
            label="Valid",
            linewidths=3,
        )
        axes[0].scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            c=(Y == 1),
            cmap="coolwarm_r",
            label="Fraud",
            linewidths=3,
        )

        axes[1].grid(True)
        axes[1].set_title("PCA Clustering", fontsize=18)
        axes[1].scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=(Y == 0),
            cmap="coolwarm_r",
            label="Valid",
            linewidths=3,
        )
        axes[1].scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=(Y == 1),
            cmap="coolwarm_r",
            label="Fraud",
            linewidths=3,
        )

        axes[2].grid(True)
        axes[2].set_title("SVD Clustering", fontsize=18)
        axes[2].scatter(
            X_svd[:, 0],
            X_svd[:, 1],
            c=(Y == 0),
            cmap="coolwarm_r",
            label="Valid",
            linewidths=3,
        )
        axes[2].scatter(
            X_svd[:, 0],
            X_svd[:, 1],
            c=(Y == 1),
            cmap="coolwarm_r",
            label="Fraud",
            linewidths=3,
        )

        st.write(figure)
else:

    st.write("<h2>Compare Models!</h2>", unsafe_allow_html=True)
    if not X_Y_data_created:

        scaled_fraud_count = df_scaled[df_scaled.Class == 1]
        scaled_valid_count = df_scaled[df_scaled.Class == 0]

        FRAUD_PER = 0.95

        num_fraud = int(FRAUD_PER * scaled_fraud_count.shape[0])

        fraud_idx = random.sample(range(0, scaled_fraud_count.shape[0]), num_fraud)
        valid_idx = random.sample(range(0, scaled_valid_count.shape[0]), num_fraud)

        fraud_subsample = scaled_fraud_count.iloc[fraud_idx]
        valid_subsample = scaled_valid_count.iloc[valid_idx]

        dataset_subsample = pd.concat([fraud_subsample, valid_subsample], axis=0)
        dataset_subsample = dataset_subsample.sample(frac=1, random_state=120)

        X, Y = create_xy(dataset_subsample, "Class")

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=120
    )

    x_train, x_test = x_train.values, x_test.values
    y_train, y_test = y_train.values, y_test.values

    st.write(
        f"* Shape of Training examples: {x_train.shape}, Shape of training labels: {y_train.shape}",
        unsafe_allow_html=True,
    )
    st.write(
        f"* Shape of Test examples: {x_test.shape}, Shape of Test labels: {y_test.shape}",
        unsafe_allow_html=True,
    )

    st.write("---", unsafe_allow_html=True)
    choice = st.sidebar.multiselect("Choose models to compare", [*avail_models.keys()])
    choice_btn = st.sidebar.button("Compare!")

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

    if choice_btn:

        figure, axes = plt.subplots(len(choice), 1, figsize=(30, 10))

        ax = {}
        for i in range(1, len(choice) + 1):
            ax.update({i: axes[i - 1]})
            count = 1

        for c in choice:
            k = c
            f = avail_models[c]
            start_time = time.time()
            _compute_result = f(x_train, y_train, x_test, y_test, gridcv_params[k])
            st.write(
                "* Time taken for {}: {:.3f}s".format(k, time.time() - start_time),
                unsafe_allow_html=True,
            )
            f, ax[count] = compare_learning_curves(
                _compute_result["gs_model"],
                x_train,
                y_train,
                (0.87, 1.01),
                f=figure,
                ax=ax[count],
                cv=cv,
                n_jobs=4,
            )
            count += 1

        st.write(f)
