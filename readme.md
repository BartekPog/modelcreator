# Automated Machine Learning Library

This library contains class **Machine** which is meant to do the **learning** for you.

## Instalation

To use the library you have to clone this repository and run:

```bash
pip install sklearn pandas joblib
```

## Usage

The library assumes that the last column of the training dataset contains the expected results. The dataset (both training and predictive) must be provided as a csv file.

If the results column contains text the _Machine_ will do its best to learn to _classify_ the data correctly. In case of a number in it _regression_ will be performed.

If the file contains headers you shall add `header_in_csv=True` parameter to the method. If you want the learning or predicting to be silent add `verbose=False`.

###### Example 1 _Iris_

```python
import automl

# Create automl machine instance
machine = automl.Machine()

# Train machine learning model
machine.learn('example-data/iris.csv')

# Predict the outcomes
machine.predict('example-data/iris-pred.csv', 'output.csv')
```

This example is also available in the `example.py` file. Consider trying it on your own.

#### DataFrame input

But what to do if a result column is not the last in the given csv? It may be inconvenient to rewrite the whole csv just to swap the columns. Because of this problem Machine has `learnFromDf` and `predictFromDf` methods. The Df in method names stands for DataFrame from pandas module. This way you can handle reading the file by yourself.

###### Example 2 _Titanic_

```python
import automl
import pandas as pd

# Create DataFrame object from file
train = pd.read_csv("titanic/train.csv")

# Get features columns from DataFrame
X_train = train.drop(['Survived'], axis=1)

# And labels (results) column
y_train = train["Survived"].astype(str)

# Create the instance of Machine
machine = automl.Machine()

# Train machine learning model
machine.learnFromDf(X_train, y_train)

# Show parameters of the model
machine.showParams()

# Load test set from fole
X_test = pd.read_csv("titanic/test.csv")

# Predict the labels
results = machine.predictFromDf(X_test)

# Save results to a new file
results.to_csv("results")
```

Simple? That's right! Just note that we used `astype(str)` in order to treat data as **classes**, not **numbers**, because the [Titanic dataset](https://www.kaggle.com/c/titanic) used in the example above has values _0_ and _1_ in `"Survived"` column to indicate whether a person made it through the disaster.

#### Saving the model

If you want your model to avoid re-learning on the whole dataset just to make a simple prediction you can save the state of _Machine_ to a file.

```python
# Save Machine with trained model to "machine.pkl"
machine.saveMachine('machine.pkl')

# Create new machine based on a schema file
machine2 = automl.Machine('machine.pkl')
```

### Development

Have a feature idea or just want to help? Take a look at the issues tab or make a pull request!
