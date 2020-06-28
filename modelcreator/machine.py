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

    def learn(self, dataset_file, header_in_csv: bool = False, metrics=None, verbose: bool = True, cv: int = 3, computationLevel: str = 'medium'):
        dataset = pd.read_csv(dataset_file, header=(
            0 if header_in_csv else None))

        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

        self.learnFromDf(X=X, y=y, metrics=metrics, verbose=verbose,
                         cv=cv, computationLevel=computationLevel)

    def learnFromDf(self, X, y, metrics=None, verbose: bool = True, cv: int = 3, computationLevel: str = 'medium'):
        self.isClassifier = isinstance(y[0], str)

        if(not isValidMetrics(metrics, self.isClassifier)):
            print("This metrics invalid")
            return None

        if(computationLevel not in computationLevels):
            print("{} is not a valid computation level".format(computationLevel))
            print("Valid computation levels:")
            for level in computationLevels:
                print("\t{}".format(level))
            return None

        self.transformer = Transformer()
        X_prep = self.transformer.fit_transform(X)

        modelData = generateModel(
            X=X_prep, y=y, isClassification=self.isClassifier, metrics=metrics, verbose=verbose, cv=cv, computationLevel=computationLevel)

        self.modelName = modelData['name']
        self.model = modelData['estimator']
        self.modelParams = modelData['params']
        self.isTrained = True

    def predict(self, features_file, output_file="output.csv", header_in_csv=False, verbose=True):
        X_pred = pd.read_csv(features_file, header=(
            0 if header_in_csv else None))

        predictions = self.predictFromDf(X_pred)

        predictions.to_csv(output_file)
        if(verbose):
            print("Results saved to ", output_file)

    def predictFromDf(self, X_predictions, output_file=None, verbose=True):
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

    def learnAndPredict(self, train_set_file, prediction_features_file,
                        output_file="output.csv", headers_in_csvs=False, metrics=None,
                        verbose=True):
        self.learn(train_set_file, headers_in_csvs, metrics, verbose)
        self.predict(prediction_features_file, output_file, headers_in_csvs)

    def saveMachine(self, output_file_name="machine.pkl"):
        with open(output_file_name, 'wb') as file:
            joblib.dump(self, file)
            print("done")

    def showParams(self):
        print(self.modelName)
        print(self.modelParams)
        print(self.model.feature_importances_)
