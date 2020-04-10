import pandas as pd
import joblib
from automl_models_generation import generateModel
from automl_pipeline import Transformer


class Machine:
    # Machine for machine learning purposes

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

    def learn(self, dataset_file, header_in_csv=False, verbose=True):
        dataset = pd.read_csv(dataset_file, header=(
            0 if header_in_csv else None))

        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

        self.learnFromDf(X, y, verbose)

    def learnFromDf(self, X, y, verbose=True):
        self.isClassifier = isinstance(y[0], str)

        self.transformer = Transformer()
        X_prep = self.transformer.fit_transform(X)

        modelData = generateModel(X_prep, y, self.isClassifier, verbose)

        self.modelName = modelData['name']
        self.model = modelData['estimator']
        self.modelParams = modelData['params']
        self.isTrained = True

    def predict(self, features_file, output_file="output.csv", header_in_csv=False):
        X_pred = pd.read_csv(features_file, header=(
            0 if header_in_csv else None))

        predictions = self.predictFromDf(X_pred)

        predictions.to_csv(output_file)
        print("Results saved to ", output_file)
        # if not self.isTrained:
        #     print("Run learn function first")
        #     return

        # X_pred = pd.read_csv(features_file, header=(
        #     0 if header_in_csv else None))

        # X_pred_prepared = self.transformer.transform(X_pred)

        # y_pred = self.model.predict(X_pred_prepared)

        # y_pred_dataframe = pd.DataFrame(data=y_pred, columns=["results"])

        # y_pred_dataframe.to_csv(output_file)
        # print("Results saved to ", output_file)

    def predictFromDf(self, X_predictions, output_file=None):
        if not self.isTrained:
            print("Run learning function first")
            return pd.DataFrame({'err': True})

        X_pred_prepared = self.transformer.transform(X_predictions)

        y_pred = self.model.predict(X_pred_prepared)

        y_pred_dataframe = pd.DataFrame(data=y_pred, columns=["results"])

        if(isinstance(output_file, str)):
            y_pred_dataframe.to_csv(output_file)
            print("Results saved to ", output_file)

        return y_pred_dataframe

    def learnAndPredict(self, train_set_file, prediction_features_file,
                        output_file="output.csv", headers_in_csvs=False,
                        verbose=True):
        self.learn(train_set_file, headers_in_csvs, verbose)
        self.predict(prediction_features_file, output_file, headers_in_csvs)

    # def predictOne(self):
    #     pass

    def saveMachine(self, output_file_name="machine.pkl"):
        with open(output_file_name, 'wb') as file:
            joblib.dump(self, file)

    def showParams(self):
        print(self.modelName)
        print(self.modelParams)
        print(self.model.feature_importances_)
