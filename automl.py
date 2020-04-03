import pandas as pd
import joblib
from automl_models_generation import generateModel
from automl_pipeline import Transformer


class Machine:
    # Machine for machine learning purposes

    def __init__(self, schema=None):
        if isinstance(schema, str):
            self = joblib.load(schema)
        else:
            self.modelParams = None
            self.model = None
            self.transformer = None
            self.isClassifier = False
            self.isTrained = False

    def learn(self, dataset_file, header_in_csv=False, verbose=True):
        dataset = pd.read_csv(dataset_file, header=(
            0 if header_in_csv else None))

        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

        self.isClassifier = isinstance(y[0], str)

        self.transformer = Transformer()
        X_prep = self.transformer.fit_transform(X)

        modelData = generateModel(X_prep, y, self.isClassifier, verbose)

        self.modelName = modelData['name']
        self.model = modelData['estimator']
        self.modelParams = modelData['params']
        self.isTrained = True

    def predict(self, features_file, output_file="output.csv", header_in_csv=False):
        if not self.isTrained:
            print("Run learn function first")
            return

        X_pred = pd.read_csv(features_file, header=(
            0 if header_in_csv else None))

        X_pred_prepared = self.transformer.transform(X_pred)

        y_pred = self.model.predict(X_pred_prepared)

        y_pread_dataframe = pd.DataFrame(data=y_pred, columns=["results"])

        y_pread_dataframe.to_csv(output_file)
        print("Results saved to ", output_file)

    def learnAndPredict(self, train_set_file, prediction_features_file,
                        output_file="output.csv", headers_in_csvs=False,
                        verbose=True):
        self.learn(train_set_file, headers_in_csvs, verbose)
        self.predict(prediction_features_file, output_file, headers_in_csvs)

    # def predictOne(self):
    #     pass

    def saveMachine(self, outputPath="machine.pkl"):
        joblib.dump(self, outputPath)

    def showParams(self):
        print(self.modelName)
        print(self.modelParams)
        print(self.model.feature_importances_)
