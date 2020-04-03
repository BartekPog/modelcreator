from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class __DataFrameSelectorByIdx(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_ids):
        self.attribute_ids = attribute_ids

    def fit(self, X, y=None):
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
