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

logreg = LogisticRegression(max_iter=200)
svc = SVC(max_iter=200)
knn = KNeighborsClassifier()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()


@st.cache(suppress_st_warning=True)
def tsne_cache(X: pd.DataFrame):
    st.write("cache miss tsne")
    return TSNE(n_components=2, random_state=42).fit_transform(X.values)


@st.cache
def pca_cache(X: pd.DataFrame):
    return PCA(n_components=2, random_state=42).fit_transform(X.values)


@st.cache
def truncated_svd_cache(X: pd.DataFrame):
    return TruncatedSVD(
        n_components=2, algorithm="randomized", random_state=42
    ).fit_transform(X.values)


@st.cache
def logreg_cache(x_train, y_train, x_test, y_test, gridcv_params):
    _res = {}
    start_time = time.time()
    logreg.fit(x_train, y_train)
    end_time = time.time()

    score = cross_val_score(logreg, x_train, y_train, cv=5)

    _res.update(
        {
            "model_name": logreg.__class__.__name__,
            "vanilla_time": end_time - start_time,
            "vanilla_cv_score": round(score.mean(), 2) * 100,
        }
    )

    _grid_search_model = GridSearchCV(logreg, gridcv_params)
    _grid_search_model.fit(x_train, y_train)
    _gs_model_estimator = _grid_search_model.best_estimator_

    _gs_score = cross_val_score(_gs_model_estimator, x_train, y_train, cv=5)
    _gs_pred = _gs_model_estimator.predict(x_test)
    _gs_acc_score = accuracy_score(y_test, _gs_pred)

    _res.update(
        {
            "gs_acc": round(_gs_acc_score.mean(), 2) * 100,
            "gs_cross_val": round(_gs_acc_score.mean(), 2) * 100,
            "gs_model": _gs_model_estimator,
        }
    )

    return _res


@st.cache
def knn_cache(x_train, y_train, x_test, y_test, gridcv_params):
    _res = {}
    start_time = time.time()
    knn.fit(x_train, y_train)
    end_time = time.time()

    score = cross_val_score(knn, x_train, y_train, cv=5)

    _res.update(
        {
            "model_name": knn.__class__.__name__,
            "vanilla_time": end_time - start_time,
            "vanilla_cv_score": round(score.mean(), 2) * 100,
        }
    )

    _grid_search_model = GridSearchCV(knn, gridcv_params)
    _grid_search_model.fit(x_train, y_train)
    _gs_model_estimator = _grid_search_model.best_estimator_

    _gs_score = cross_val_score(_gs_model_estimator, x_train, y_train, cv=5)
    _gs_pred = _gs_model_estimator.predict(x_test)
    _gs_acc_score = accuracy_score(y_test, _gs_pred)

    _res.update(
        {
            "gs_acc": round(_gs_acc_score.mean(), 2) * 100,
            "gs_cross_val": round(_gs_acc_score.mean(), 2) * 100,
            "gs_model": _gs_model_estimator,
        }
    )

    return _res


@st.cache
def svc_cache(x_train, y_train, x_test, y_test, gridcv_params):
    _res = {}
    start_time = time.time()
    svc.fit(x_train, y_train)
    end_time = time.time()

    score = cross_val_score(svc, x_train, y_train, cv=5)

    _res.update(
        {
            "model_name": svc.__class__.__name__,
            "vanilla_time": end_time - start_time,
            "vanilla_cv_score": round(score.mean(), 2) * 100,
        }
    )

    _grid_search_model = GridSearchCV(svc, gridcv_params)
    _grid_search_model.fit(x_train, y_train)
    _gs_model_estimator = _grid_search_model.best_estimator_

    _gs_score = cross_val_score(_gs_model_estimator, x_train, y_train, cv=5)
    _gs_pred = _gs_model_estimator.predict(x_test)
    _gs_acc_score = accuracy_score(y_test, _gs_pred)

    _res.update(
        {
            "gs_acc": round(_gs_acc_score.mean(), 2) * 100,
            "gs_cross_val": round(_gs_acc_score.mean(), 2) * 100,
            "gs_model": _gs_model_estimator,
        }
    )

    return _res


@st.cache
def dtc_cache(x_train, y_train, x_test, y_test, gridcv_params):
    _res = {}
    start_time = time.time()
    dtc.fit(x_train, y_train)
    end_time = time.time()

    score = cross_val_score(dtc, x_train, y_train, cv=5)

    _res.update(
        {
            "model_name": dtc.__class__.__name__,
            "vanilla_time": end_time - start_time,
            "vanilla_cv_score": round(score.mean(), 2) * 100,
        }
    )

    _grid_search_model = GridSearchCV(dtc, gridcv_params)
    _grid_search_model.fit(x_train, y_train)
    _gs_model_estimator = _grid_search_model.best_estimator_

    _gs_score = cross_val_score(_gs_model_estimator, x_train, y_train, cv=5)
    _gs_pred = _gs_model_estimator.predict(x_test)
    _gs_acc_score = accuracy_score(y_test, _gs_pred)

    _res.update(
        {
            "gs_acc": round(_gs_acc_score.mean(), 2) * 100,
            "gs_cross_val": round(_gs_acc_score.mean(), 2) * 100,
            "gs_model": _gs_model_estimator,
        }
    )

    return _res


@st.cache
def rfc_cache(x_train, y_train, x_test, y_test, gridcv_params):
    _res = {}
    start_time = time.time()
    rfc.fit(x_train, y_train)
    end_time = time.time()

    score = cross_val_score(rfc, x_train, y_train, cv=5)

    _res.update(
        {
            "model_name": rfc.__class__.__name__,
            "vanilla_time": end_time - start_time,
            "vanilla_cv_score": round(score.mean(), 2) * 100,
        }
    )

    _grid_search_model = GridSearchCV(rfc, gridcv_params)
    _grid_search_model.fit(x_train, y_train)
    _gs_model_estimator = _grid_search_model.best_estimator_

    _gs_score = cross_val_score(_gs_model_estimator, x_train, y_train, cv=5)
    _gs_pred = _gs_model_estimator.predict(x_test)
    _gs_acc_score = accuracy_score(y_test, _gs_pred)

    _res.update(
        {
            "gs_acc": round(_gs_acc_score.mean(), 2) * 100,
            "gs_cross_val": round(_gs_acc_score.mean(), 2) * 100,
            "gs_model": _gs_model_estimator,
        }
    )

    return _res
