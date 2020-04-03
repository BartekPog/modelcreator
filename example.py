import automl
import pandas as pd


# iris = pd.read_csv('data/iris.csv')

# X = iris.iloc[:, :-1]

# X.to_csv("data/iris-only-x.csv", index=False)

machine = automl.Machine()

machine.learnAndPredict("data/iris.csv", "data/iris-only-x.csv")

# machine.predict("data/iris-only-x.csv")

# machine.saveMachine()

machine.showParams()

# machine2 = automl.Machine("machine.pkl")

# machine2.showParams()
