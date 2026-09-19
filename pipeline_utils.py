"""
pipeline_utils.py - Shared custom sklearn transformers
=======================================================
These classes must be importable at model load time (joblib deserialises
the pipeline by looking up the class in its original module).
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select a subset of DataFrame columns and return a numpy array."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns].values


class TextSelector(BaseEstimator, TransformerMixin):
    """Select a single text column and return it as a string Series."""

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.column].fillna("").astype(str)
