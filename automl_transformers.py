from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class __DataFrameSelectorByIdx(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_ids, acceptable_missing_threshold=0.35):
        self.attribute_ids = attribute_ids
        self.acceptable_missing_threshold = acceptable_missing_threshold

    def fit(self, X, y=None):
        # Exclude columns with number of missing values exceeding threshold
        allRows = len(X)
        missing = list(X.isna().sum())
        self.attribute_ids = list(filter(
            lambda idx: missing[idx]/allRows <= self.acceptable_missing_threshold, self.attribute_ids))

        return self

    def transform(self, X):
        return X.iloc[:, self.attribute_ids].values


def NumericalPipeline(numeric_indexes):
    return Pipeline([
        ('selector', __DataFrameSelectorByIdx(numeric_indexes)),
        ('imputer', SimpleImputer(strategy="median")),
        ('min_max_scaler', MinMaxScaler()),
        #     ('std_scaler', StandardScaler())
    ])


def CategoricalPipeline(categorical_indexes):
    return Pipeline([
        ('selector', __DataFrameSelectorByIdx(categorical_indexes)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('cat_encoder', OneHotEncoder(sparse=False))
    ])
