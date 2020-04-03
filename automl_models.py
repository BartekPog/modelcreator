from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

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
    },
    {
        'name': "Ada Boost Classifier",
        'estimator': AdaBoostClassifier(),
        'params': [
            {},
            {
                'n_estimators': [20, 50, 80, 100]
            }
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
        'name': "Random Forest Regressor",
        'estimator':  RandomForestRegressor(),
        'params': [
            {},
            {
                'n_estimators': [10, 70, 100, 120, 160], 'max_depth': [None, 5, 15, 30],
                'max_features': [None, 'auto'],
            },
        ]
    },
    {
        'name': "Ada Boost Regressor",
        'estimator': AdaBoostRegressor(),
        'params': [
            {},
            {
                'n_estimators': [20, 50, 80, 100]
            }
        ]
    }
]
