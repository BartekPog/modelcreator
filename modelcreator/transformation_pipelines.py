from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from .transformation_utils import __DataFrameSelectorByIdx, __ClassReducer


def NumericalPipeline(numeric_indexes):
    return Pipeline([
        ('selector', __DataFrameSelectorByIdx(numeric_indexes)),
        ('imputer', SimpleImputer(strategy="median")),
        # ('min_max_scaler', MinMaxScaler()),
        ('std_scaler', StandardScaler())
    ])


def CategoricalPipeline(categorical_indexes):
    return Pipeline([
        ('selector', __DataFrameSelectorByIdx(
            categorical_indexes, acceptable_missing_threshold=3.35)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('class_reducer', __ClassReducer(min_class_frequency=0.05)),
        ('cat_encoder', OneHotEncoder(sparse=False))
    ])
