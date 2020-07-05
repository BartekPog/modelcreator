# modelcreator - AutoML package

This package contains a **Machine** which is meant to do the **learning** for you. It can automaticly create a fitting predictive model for given data.

###### Sample output

```
Testing:  Gradient Boosting Classifier
[########################################] | 100% Completed |  3.9s
Score: 0.9667

Testing:  Ada Boost Classifier
[########################################] | 100% Completed |  1.3s
Score: 0.9600

Testing:  Random Forest Classifier
[########################################] | 100% Completed |  5.0s
Score: 0.9600

Testing:  Balanced Random Forest Classifier
[########################################] | 100% Completed |  3.5s
Score: 0.9600

Testing:  SVC
[########################################] | 100% Completed |  1.2s
Score: 0.9667

Chosen model:  Gradient Boosting Classifier 0.9667

Params:
        min_samples_split: 2
        n_estimators: 100

Results saved to  output.csv
```

# Table of Contents

1. [Installation](#installation)
1. [Usage](#usage)
   - [CSV input](#csv-path-input)
   - [Pandas input](#pandas-input)
1. [Saving model](#saving-the-model)
1. [Parameters](#parameters)
   - [Machine](#machine)
   - [learn](#learn)
   - [learnFromDf](#learnfromdf)
   - [predict](#predict)
   - [predictFromDf](#predictfromdf)
   - [saveMachine](#savemachine)
1. [Development](#development)

## Installation

To use the package run:

```bash
pip install modelcreator
```

## Usage

The input may be either a path to a **csv** file or a **pandas DataFrame** object.

#### CSV path input

The library assumes that the last column of the training dataset contains the expected results. The dataset (both training and predictive) must be provided as a **csv** file.

If the results column contains text the _Machine_ will do its best to learn to _classify_ the data correctly. In case of a number inside, _regression_ will be performed.

If the file contains headers you shall add `header_in_csv=True` parameter to the method.

###### Example 1 _Iris_

```python
from modelcreator import Machine

# Create automl machine instance
machine = Machine()

# Train machine learning model
machine.learn('example-data/iris.csv')

# Predict the outcomes
machine.predict('example-data/iris-pred.csv', 'output.csv')
```

This example is also available in the `example.py` file. Consider trying it on your own.

#### Pandas input

But what to do if a result column is not the last in the given csv? It may be inconvenient to rewrite the whole csv just to swap the columns. Because of this problem Machine has `learnFromDf` and `predictFromDf` methods. The _Df_ in method names stands for _DataFrame_ from _pandas_ module. This way you can handle reading the file by yourself.

###### Example 2 _Titanic_

```python
from modelcreator import Machine
import pandas as pd

# Create DataFrame object from file
train = pd.read_csv("train.csv")

# Get features columns from DataFrame
X_train = train.drop(['Survived'], axis=1)

# And labels (results) column
y_train = train["Survived"].astype(str)

# Create the instance of Machine
machine = Machine()

# Train machine learning model
machine.learnFromDf(X_train, y_train, computation_level='advanced')

# Show parameters of the model
machine.showParams()

# Load test set from file
X_test = pd.read_csv("test.csv")

# Predict the labels
results = machine.predictFromDf(X_test)

# Save results to a new file
results.to_csv("results.csv")
```

Simple? That's right! Just note that we used `astype(str)` in order to treat data as **classes**, not **numbers** because the [Titanic dataset](https://www.kaggle.com/c/titanic) used in the example above has values _0_ and _1_ in `"Survived"` column to indicate whether a person made it through the disaster.

#### Saving the model

If you want your model to avoid re-learning on the whole dataset just to make a simple prediction you can save the state of _Machine_ to a file.

```python
# Save Machine with a trained model to "machine.pkl"
machine.saveMachine('machine.pkl')

# Create a new machine based on a schema file
machine2 = Machine('machine.pkl')
```

#### Parameters

The **Machine** can be customized according to the use case. Check the parameters table:

###### Machine

| Param  | Type            | Default | Description                                                                                                                                           |
| ------ | --------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| schema | _None_ or _str_ | `None`  | A Machine may be created based on a saved, pre-trained machine instance. You may specify the path to the saved instance in this param to recreate it. |

###### learn

| Param             | Type                        | Default                                         | Description                                                                                                                                                                                                                               |
| ----------------- | --------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| dataset_file      | _str_                       |                                                 | Path to a csv file which contains training dataset.                                                                                                                                                                                       |
| header_in_csv     | _bool_                      | `False`                                         | Whether the csv file contains _headers_ in the first row.                                                                                                                                                                                 |
| metrics           | _None_, _str_ or _Callable_ | `'accuracy'` or `'neg_root_mean_squared_error'` | Metrics used for scoring estimators. Many popular scoring functions (such as _f1_, _roc_auc_, _neg_mean_gamma_deviance_). See [here](https://scikit-learn.org/stable/modules/model_evaluation.html) how to make custom scoring functions. |
| verbose           | _bool_                      | `True`                                          | Whether to print learning logs.                                                                                                                                                                                                           |
| cv                | _int_                       | `3`                                             | a Number of cross-validation subsets. Higher values may increase computation time.                                                                                                                                                        |
| computation_level | _str_                       | `'medium'`                                      | Can be either `'basic'`, `'medium'` or `'advanced'`. With higher computation level more models and parameters are being tested.                                                                                                           |

###### learnFromDf

| Param             | Type                        | Default                                         | Description                                                                                                                                                                                                                               |
| ----------------- | --------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| X                 | _pandas.DataFrame_          |                                                 | DataFrame containing the feature columns.                                                                                                                                                                                                 |
| y                 | _pandas.Series_             |                                                 | Label columns of the training data.                                                                                                                                                                                                       |
| metrics           | _None_, _str_ or _Callable_ | `'accuracy'` or `'neg_root_mean_squared_error'` | Metrics used for scoring estimators. Many popular scoring functions (such as _f1_, _roc_auc_, _neg_mean_gamma_deviance_). See [here](https://scikit-learn.org/stable/modules/model_evaluation.html) how to make custom scoring functions. |
| verbose           | _bool_                      | `True`                                          | Whether to print learning logs.                                                                                                                                                                                                           |
| cv                | _int_                       | `3`                                             | A number of cross-validation subsets. Higher values may increase computation time.                                                                                                                                                        |
| computation_level | _str_                       | `'medium'`                                      | Can be either `'basic'`, `'medium'` or `'advanced'`. With higher computation level more models and parameters are being tested.                                                                                                           |

###### predict

| Param         | Type   | Default        | Description                                                                   |
| ------------- | ------ | -------------- | ----------------------------------------------------------------------------- |
| features_file | _str_  |                | Path to the features **csv** of the data to generate predictions on.          |
| header_in_csv | _bool_ | `False`        | Whether the csv file contains _headers_ in the first row.                     |
| output_file   | _str_  | `'output.csv'` | Path to the output **csv** file. In this file, the predictions will be saved. |
| verbose       | _str_  | `True`         | Whether to print logs.                                                        |

###### predictFromDf

| Param         | Type               | Default | Description                                                                                                                                                                                                                          |
| ------------- | ------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| X_predictions | _pandas.DataFrame_ |         | Features columns to generate predictions on.                                                                                                                                                                                         |
| output_file   | _str_              | `None`  | Predict method returns _pandas.Series_ of the results. Additionally, it can also save the results to a **csv** file. It can be specified here. If the path is other than `None` it will be interpreted as a path to the output file. |
| verbose       | _str_              | `True`  | Whether to print logs.                                                                                                                                                                                                               |

###### saveMachine

| Param            | Type  | Default         | Description                                        |
| ---------------- | ----- | --------------- | -------------------------------------------------- |
| output_file_name | _str_ | `'machine.pkl'` | Path to where shall the Machine instance be saved. |

### Development

Have a feature idea or just want to help? Take a look at the [issues tab](https://github.com/BartekPog/modelcreator/issues)!
