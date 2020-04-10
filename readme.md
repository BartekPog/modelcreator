# Automated Machine Learning Library

This library contains class **Machine** which is meant to do the **learning** for you.

### Instalation

To use the library you have to clone this repositorium and run:

```bash
pip install sklearn pandas joblib
```

### Usage

The library assumes that the last column of the training dataset contains the expected results. The dataset (both training and predictive) must be provided as a csv file. If the file contains headers you shall add `header_in_csv=True` parameter to the method. If you want the learning or predicting to be silent add `verbose=False`.

###### Example

```python
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
```

This example is also available in the `example.py` file. Consider trying it on your own.

### TODO

- [x] Create main files structure and test script on Iris dataset,
- [x] Handle sparse data (eg. columns with less than 60% data shall be dropped),
- [x] Test on other easy datasets,
- [x] Add class methods with input as pandas dataset instead of csv,
- [ ] Handle too many class variables (by replacing uncommon as "rare"),
- [ ] Test on harder datasets,
- [ ] Expand Machine model report printing function,
- [ ] Expand documentation,
- [ ] Add unit tests,
- [ ] Add more models and parameters to the grid,
- [ ] Add gif to readme,
