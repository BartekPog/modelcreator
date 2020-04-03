import automl
import pandas as pd

# Create automl machine instance
machine = automl.Machine()

# Train machine learning model and make a prediction
machine.learnAndPredict("data/iris.csv", "data/iris-only-x.csv")

# Show parameters of the model
machine.showParams()

# Save Machine instance to "machine.pkl"
machine.saveMachine("machine.pkl")

# Create new machine based on the schema
machine2 = automl.Machine("machine.pkl")
