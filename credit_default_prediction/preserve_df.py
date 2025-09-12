import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PreserveDF(BaseEstimator, TransformerMixin):
    def __init__(self, transformer: TransformerMixin):
        self.transformer = transformer
        self.columns_ = None

    def fit(self, X: pd.DataFrame, y=None):
        self.transformer.fit(X, y)
        # Try to get output feature names safely
        try:
            self.columns_ = self.transformer.get_feature_names_out()
        except (AttributeError, NotImplementedError):
            # Fall back to input columns if not available
            self.columns_ = getattr(X, "columns", None)
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X_t = self.transformer.transform(X)
        return pd.DataFrame(X_t, columns=self.columns_, index=X.index)
