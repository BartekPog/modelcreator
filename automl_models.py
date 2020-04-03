from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

classificationModels = [
    {
        'name': "Gradient Boosting Classifier",
        'estimator': GradientBoostingClassifier(),
        'params': [
                {},
                {
                    'n_estimators': [70, 100, 150, 240], 'min_samples_split': [2, 3, 4],
                    'max_depth': [2, 3, 4], 'max_features': [None, 'auto']
                }
        ]
    },
    {
        'name': "Random Forest Classifier",
        'estimator':  RandomForestClassifier(),
        'params': [
                {},
                {
                    'n_estimators': [70, 100, 120, 160], 'max_depth': [None, 5, 15, 30],
                    'max_features': [None, 'auto']
                },
        ]
    }
]
regressionModels = [
    {
        'name': "Gradient Boosting Regressor",
        'estimator': GradientBoostingRegressor(),
        'params': [
                {},
                {
                    'n_estimators': [70, 100, 150, 240], 'min_samples_split': [2, 3, 4],
                    'max_depth': [2, 3, 4], 'max_features': [None, 'auto'], 'loss':['ls', 'huber']
                }
        ]
    },
    {
        'name': "Random Forest Classifier",
        'estimator':  RandomForestRegressor(),
        'params': [
                {},
                {
                    'n_estimators': [10, 70, 100, 120, 160], 'max_depth': [None, 5, 15, 30],
                    'max_features': [None, 'auto'],
                },
        ]
    }
]
