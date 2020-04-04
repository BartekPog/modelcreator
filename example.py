import automl

# Create automl machine instance
machine = automl.Machine()

# Train machine learning model
machine.learn('example-data/iris.csv')

# Predict the outcomes
machine.predict('example-data/iris-pred.csv', 'output.csv')

# Show parameters of the model
machine.showParams()

# Save Machine with trained model to "machine.pkl"
machine.saveMachine('machine.pkl')

# Create new machine based on the schema
machine2 = automl.Machine('machine.pkl')
