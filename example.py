from modelcreator import Machine

# Create automl machine instance
machine = Machine()

# Train machine learning model
machine.learn('example-data/iris.csv')

# Predict the outcomes
machine.predict('example-data/iris-pred.csv', 'output.csv')
