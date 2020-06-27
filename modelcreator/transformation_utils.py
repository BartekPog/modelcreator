from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


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


class __ClassReducer(BaseEstimator, TransformerMixin):
    def __init__(self, min_class_frequency=0.05):
        self.min_class_frequency = min_class_frequency

    def fit(self, X, y=None):
        self.possibleValues = []

        threshold = len(X)*self.min_class_frequency

        for column in X.T:
            notRareInCol = set()
            colNoNan = ["" if x is np.nan else x for x in column]

            for unique, counts in np.column_stack(np.unique(colNoNan, return_counts=True)):
                if(counts.astype(int) > threshold):
                    notRareInCol.add(unique)

            notRareInCol.discard("")
            self.possibleValues.append(notRareInCol)

        return self

    def transform(self, X):
        newX = X.T
        for colIdx, valuesSet in enumerate(self.possibleValues, 0):
            newX[colIdx] = [x if (x in valuesSet)
                            else "RARE" for x in newX[colIdx]]
        return newX.T
