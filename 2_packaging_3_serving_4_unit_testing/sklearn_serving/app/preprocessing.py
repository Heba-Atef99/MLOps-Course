import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DomainProcessing(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_sum: list[str] | None = None):
        self.columns_to_sum = columns_to_sum or []

    def fit(self, X: pd.DataFrame, y: object = None) -> "DomainProcessing":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["total_assets_value"] = X[self.columns_to_sum].sum(axis=1)
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop: list[str] | None = None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X: pd.DataFrame, y: object = None) -> "DropColumns":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns_to_drop)


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoding_map: dict[str, list[str]] | None = None):
        self.encoding_map = encoding_map or {}

    def fit(self, X: pd.DataFrame, y: object = None) -> "CustomLabelEncoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, positive_values in self.encoding_map.items():
            pv = positive_values
            X[col] = X[col].apply(lambda x, _pv=pv: 1 if str(x).strip() in _pv else 0)
        return X


class LogTransforms(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or []

    def fit(self, X: pd.DataFrame, y: object = None) -> "LogTransforms":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            X[col] = np.log(X[col].clip(lower=1))
        return X
