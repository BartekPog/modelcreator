import joblib
import automl_models
import automl_transformers


class Machine:
    # Machine for machine learning purposes

    def __init__(self, schema=None):
        if isinstance(schema, str):
            self = joblib.load(schema)
        else:
            self.params = None
            self.model = None
            self.transformers = None

    def learn(self, dataset_path, header_in_csv=False, verbose=True, saveParams=False):
        # creates a model and transformer based on a csv file
        pass

    def predict(self):
        pass

    def learnAndPredict(self):
        pass

    def predictOne(self):
        pass

    def saveMachine(self, outputPath="machine.pkl"):
        joblib.dump(self, outputPath)

    def showParams(self):
        pass
