import pandas as pd
import joblib
from .models_generation import generateModel
from .transformer import Transformer
from .metrics import isValidMetrics
from .models import computationLevels


class Machine:
    def __init__(self, schema=None):
        self.modelParams = None
        self.modelName = "There is no model yet"
        self.model = None
        self.transformer = None
        self.isClassifier = False
        self.isTrained = False
        if isinstance(schema, str):
            with open(schema, 'rb') as file:
                prev = joblib.load(file)
                self.modelParams = prev.modelParams
                self.modelName = prev.modelName
                self.model = prev.model
                self.transformer = prev.transformer
                self.isClassifier = prev.isClassifier
                self.isTrained = prev.isTrained

    def learn(self, dataset_file: pd.DataFrame, header_in_csv: bool = False, metrics=None, verbose: bool = True, cv: int = 3, computation_level: str = 'medium'):
        dataset = pd.read_csv(dataset_file, header=(
            0 if header_in_csv else None))

        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

        self.learnFromDf(X=X, y=y, metrics=metrics, verbose=verbose,
                         cv=cv, computation_level=computation_level)

    def learnFromDf(self, X: pd.DataFrame, y: pd.Series, metrics=None, verbose: bool = True, cv: int = 3, computation_level: str = 'medium'):
        self.isClassifier = isinstance(y[0], str)

        if(not isValidMetrics(metrics, self.isClassifier)):
            print("This metrics invalid")
            return None

        if(computation_level not in computationLevels):
            print("{} is not a valid computation level".format(computation_level))
            print("Valid computation levels:")
            for level in computationLevels:
                print("\t{}".format(level))
            return None

        self.transformer = Transformer()
        X_prep = self.transformer.fit_transform(X)

        modelData = generateModel(
            X=X_prep, y=y, isClassification=self.isClassifier, metrics=metrics, verbose=verbose, cv=cv, computationLevel=computation_level)

        self.modelName = modelData['name']
        self.model = modelData['estimator']
        self.modelParams = modelData['params']
        self.isTrained = True

    def predict(self, features_file: str, output_file="output.csv", header_in_csv: bool = False, verbose: bool = True):
        X_pred = pd.read_csv(features_file, header=(
            0 if header_in_csv else None))

        predictions = self.predictFromDf(X_pred)

        predictions.to_csv(output_file)
        if(verbose):
            print("Results saved to ", output_file)

    def predictFromDf(self, X_predictions: pd.DataFrame, output_file: str = None, verbose: bool = True):
        if not self.isTrained:
            print("Run learning function first")
            return pd.DataFrame({'err': True})

        X_pred_prepared = self.transformer.transform(X_predictions)

        y_pred = self.model.predict(X_pred_prepared)

        y_pred_dataframe = pd.DataFrame(data=y_pred, columns=["results"])

        if(isinstance(output_file, str)):
            y_pred_dataframe.to_csv(output_file)
            if(verbose):
                print("Results saved to ", output_file)

        return y_pred_dataframe

    def saveMachine(self, output_file_name: str = "machine.pkl"):
        with open(output_file_name, 'wb') as file:
            joblib.dump(self, file)
            print("done")

    def showParams(self):
        print("Model: ", self.modelName)

        print("\nParams:")
        params = self.modelParams

        if(len(params) > 0):
            for key, value in params.items():
                print("\t{}: {}".format(key, value))

        else:
            print("\tdefault")

        print("")
