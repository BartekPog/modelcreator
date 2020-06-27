import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from .transformation_pipelines import NumericalPipeline, CategoricalPipeline


class Transformer:
    def __init__(self):
        self.inputVariables = 0
        self.numeric_ids = []
        self.categorical_ids = []
        self.ready = False

    def fit_transform(self, X):
        self.inputVariables = len(X.columns)

        for idx, dtype in enumerate(X.dtypes):
            if pd.api.types.is_numeric_dtype(dtype):
                self.numeric_ids.append(idx)
            else:
                self.categorical_ids.append(idx)

        transformers = []

        if len(self.numeric_ids) > 0:
            transformers.append(
                ("numerical_pipeline", NumericalPipeline(self.numeric_ids)))

        if len(self.categorical_ids) > 0:
            transformers.append(
                ("categorical_pipeline", CategoricalPipeline(self.categorical_ids)))

        self.pipeline = FeatureUnion(transformer_list=transformers)

        self.ready = True
        return self.pipeline.fit_transform(X)

    def transform(self, X):
        if(not self.ready):
            print("Cannot transform. You must call 'fit_transform' first")
            return None

        if (len(X.columns) != self.inputVariables):
            print("Column number must be equal in training and predicting process")
            return None

        return self.pipeline.transform(X)

    def getIds(self):
        return {
            'all': self.inputVariables,
            'categorical': self.categorical_ids,
            'numerical': self.numeric_ids
        }
