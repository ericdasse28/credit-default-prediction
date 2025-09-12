import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PreserveDF(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer
        self.columns = None

    def fit(self, X: pd.DataFrame, y=None):
        self.columns = X.columns
        self.transformer.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X_t = self.transformer.transform(X)
        return pd.DataFrame(X_t, columns=self.columns, index=X.index)
