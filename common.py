"""
Shared pieces of the MLR Exercise 6 e-bike experiment ("Will the e-bike
finish the trip?").

Everything here is copied from ex06_notebook.ipynb (Tasks 4a-4c) with the
behavior preserved, so that the standalone comparison in bayes_compare.py
runs on exactly the same pipeline the notebook established:

* data loading that prefers the local ``ex06_ebikes.csv`` and falls back to
  the public course URL (notebook Task 4a loader),
* the fixed physical cleaning rules ``clean_readings`` (invalid charge_pct
  and distance_km become NaN; valid extremes are kept),
* the engineered feature functions ``add_energy`` and the chosen extension
  ``add_extended_features`` (available_wh, then wh_per_km),
* the untouched baseline pipeline: physical cleaning -> numeric median
  imputation with identity features + categorical "missing" label and
  one-hot encoding -> RandomForestClassifier(n_estimators=100,
  random_state=0),
* the fixed stratified train/test split and the fixed CV splitter.

Nothing in this module is fitted, tuned or searched; the constants are the
student's choices from the notebook. All searching lives in
bayes_compare.py and uses only the training subset.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# --- constants from the notebook (Task 4a) ---------------------------------

DATA = "https://raw.githubusercontent.com/tomasvicar/MLR-public/master/exercises/data/"
LOCAL_CSV = Path(__file__).resolve().parent / "ex06_ebikes.csv"

NUMERIC = ["capacity_wh", "charge_pct", "battery_age_years", "distance_km",
           "ascent_m", "temperature_c", "headwind_kmh", "payload_kg"]
CATEGORICAL = ["surface", "bike_type", "assistance"]
TARGET = "runs_out"

# All feature and model selection happens on the training subset only,
# always through this fixed splitter.
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)


def load_ebikes():
    """Load the e-bike table (same loader as the notebook Task 4a cell).

    The local CSV shipped with the notebook is preferred; the public raw
    course URL is the fallback. The local path is resolved next to this
    module, so the loader works from any working directory.
    """
    return pd.read_csv(LOCAL_CSV if LOCAL_CSV.exists() else DATA + "ex06_ebikes.csv")


def split_data(ebikes):
    """Fixed stratified split from the notebook: test_size=0.25, random_state=42.

    X_test / y_test stay untouched until final reporting.
    """
    X = ebikes[NUMERIC + CATEGORICAL]
    y = ebikes[TARGET]
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


# --- fixed cleaning and feature functions (notebook Tasks 4a/4c) -----------

def clean_readings(frame):
    """Copy the frame; replace physically impossible readings with NaN.

    Charge outside [0, 100] % and distance <= 0 km are impossible. Other
    extreme values are legitimate trips and are kept.
    """
    frame = frame.copy()
    frame.loc[(frame["charge_pct"] < 0) | (frame["charge_pct"] > 100), "charge_pct"] = np.nan
    frame.loc[frame["distance_km"] <= 0, "distance_km"] = np.nan
    return frame


def add_energy(frame):
    """Numeric features after fold-internal imputation: add available_wh."""
    res = frame.copy()
    res["available_wh"] = res["capacity_wh"] * res["charge_pct"] / 100.0
    return res


def add_extended_features(frame):
    """Chosen extension from notebook Task 4c: energy budget wh_per_km.

    Keeps all original columns plus available_wh; does not modify its input.
    Ratios are safe here because invalid distances became NaN and the
    fold-fitted imputer replaced them with positive distances.
    """
    res = add_energy(frame)
    res["wh_per_km"] = res["available_wh"] / res["distance_km"]
    return res


# --- pipeline builders (same step names as the notebook) -------------------

def build_numeric_branch(impute="median", feature_func=None, scale=False):
    """Numeric branch: imputation -> optional feature creation -> optional scaling.

    With the defaults this is exactly the notebook's numeric branch
    (median imputation, identity features). StandardScaler, when requested,
    is placed *after* numeric feature creation and is always fitted inside
    each fold by cross_val_score.
    """
    steps = [("imputer", SimpleImputer(strategy=impute))]
    if feature_func is not None:
        steps.append(("features", FunctionTransformer(feature_func)))
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps).set_output(transform="pandas")


def build_categorical_branch():
    """Categorical branch: "missing" label imputation, then one-hot encoding."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def build_preprocessor(impute="median", feature_func=None, scale=False):
    """ColumnTransformer with the notebook's 'num' and 'cat' branches."""
    return ColumnTransformer(transformers=[
        ("num", build_numeric_branch(impute, feature_func, scale), NUMERIC),
        ("cat", build_categorical_branch(), CATEGORICAL),
    ])


def build_pipeline(model, impute="median", feature_func=None, scale=False):
    """Full pipeline clean -> prep -> model, with the notebook's step names."""
    return Pipeline(steps=[
        ("clean", FunctionTransformer(clean_readings)),
        ("prep", build_preprocessor(impute, feature_func, scale)),
        ("model", model),
    ])


def build_baseline():
    """The unchanged notebook baseline: original features, median imputation,
    RandomForestClassifier(n_estimators=100, random_state=0), no scaling."""
    return build_pipeline(
        model=RandomForestClassifier(n_estimators=100, random_state=0),
        impute="median",
        feature_func=None,
        scale=False,
    )


def get_experiment():
    """Return (X_train, X_test, y_train, y_test) from the fixed split."""
    return split_data(load_ebikes())
