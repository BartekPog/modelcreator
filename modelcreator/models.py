from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import Lasso, LinearRegression

computationLevels = ['basic', 'medium', 'advanced']

allModels = {
    'classification': {
        'basic': [
            {
                'name': "Gradient Boosting Classifier",
                'estimator': GradientBoostingClassifier(),
                'params': [
                    {},
                ]
            },
            {
                'name': "Random Forest Classifier",
                'estimator':  RandomForestClassifier(),
                'params': [
                    {'class_weight': [None, 'balanced']},
                ]
            },
            {
                'name': "SVC",
                'estimator': SVC(),
                'params': [
                    {'degree': [1, 3]},
                ]
            }
        ],
        'medium': [
            {
                'name': "Gradient Boosting Classifier",
                'estimator': GradientBoostingClassifier(),
                'params': [
                    {
                        'n_estimators': [100, 150], 'min_samples_split': [2, 4],
                    }
                ]
            },
            {
                'name': "Ada Boost Classifier",
                'estimator': AdaBoostClassifier(),
                'params': [
                    {
                        'n_estimators': [20, 50, 80, 100]
                    }
                ]
            },
            {
                'name': "Random Forest Classifier",
                'estimator':  RandomForestClassifier(),
                'params': [
                    {
                        'n_estimators': [100, 160],
                        'max_features': [None, 'auto'], 'class_weight': [None, 'balanced']
                    },
                ]
            },
            {
                'name': "Balanced Random Forest Classifier",
                'estimator':  BalancedRandomForestClassifier(),
                'params': [
                    {
                        'max_features': [None, 'auto'], 'class_weight': [None, 'balanced']
                    },
                ]
            },
            {
                'name': "SVC",
                'estimator': SVC(),
                'params': [
                    {
                        'C': [0.7, 1], 'degree':[1, 2, 3],
                        'class_weight':[None, 'balanced']
                    },
                ]
            }
        ],
        'advanced': [
            {
                'name': "Gradient Boosting Classifier",
                'estimator': GradientBoostingClassifier(),
                'params': [
                    {
                        'n_estimators': [70, 100, 150], 'min_samples_split': [2, 3, 4],
                        'max_depth': [2, 3, 4], 'max_features': [None, 'auto']
                    }
                ]
            },
            {
                'name': "Random Forest Classifier",
                'estimator':  RandomForestClassifier(),
                'params': [
                    {
                        'n_estimators': [70, 100, 160], 'max_depth': [None, 5, 15],
                        'max_features': [None, 'auto']
                    },
                ]
            },
            {
                'name': "Balanced Random Forest Classifier",
                'estimator':  BalancedRandomForestClassifier(),
                'params': [
                    {
                        'n_estimators': [70, 100, 160], 'max_depth': [None, 5, 15],
                        'max_features': [None, 'auto'], 'class_weight': [None, 'balanced']
                    },
                ]
            },
            {
                'name': "Ada Boost Classifier",
                'estimator': AdaBoostClassifier(),
                'params': [
                    {
                        'n_estimators': [20, 50, 100, 160]
                    }
                ]
            },
            {
                'name': "SVC",
                'estimator': SVC(),
                'params': [
                    {
                        'C': [0.5, 1, 1.2, 1.4], 'degree':[1, 2, 3, 4],
                        'class_weight':[None, 'balanced']
                    },
                ]
            }
        ],
    },
    'regression': {
        'basic': [
            {
                'name': "Gradient Boosting Regressor",
                'estimator': GradientBoostingRegressor(),
                'params': [
                    {},
                ]
            },
            {
                'name': "Random Forest Regressor",
                'estimator':  RandomForestRegressor(),
                'params': [
                    {},
                ]
            },
            {
                'name': "SVR",
                'estimator': SVR(),
                'params': [
                    {},
                ]
            },
            {
                'name': "Linear Regression",
                'estimator': LinearRegression(),
                'params': [
                    {},
                ]
            },
        ],
        'medium': [
            {
                'name': "Gradient Boosting Regressor",
                'estimator': GradientBoostingRegressor(),
                'params': [
                    {
                        'n_estimators': [70, 100], 'min_samples_split': [2, 4],
                        'max_depth': [2,  4],
                    }
                ]
            },
            {
                'name': "Random Forest Regressor",
                'estimator':  RandomForestRegressor(),
                'params': [
                    {
                        'n_estimators': [30, 100], 'max_depth': [None, 5],
                    },
                ]
            },
            {
                'name': "Ada Boost Regressor",
                'estimator': AdaBoostRegressor(),
                'params': [
                    {
                        'n_estimators': [20, 50, 100, 120]
                    }
                ]
            },
            {
                'name': "SVR",
                'estimator': SVR(),
                'params': [
                    {'degree': [2, 3], 'C':[0.6, 1]},
                ]
            },
            {
                'name': "Lasso",
                'estimator': Lasso(),
                'params': [
                    {'alpha': [0.5, 1]},
                ]
            },
            {
                'name': "Linear Regression",
                'estimator': LinearRegression(),
                'params': [
                    {},
                ]
            },
        ],
        'advanced': [
            {
                'name': "Gradient Boosting Regressor",
                'estimator': GradientBoostingRegressor(),
                'params': [
                    {
                        'n_estimators': [70, 100, 150], 'min_samples_split': [2, 3, 4],
                        'max_depth': [2,  4], 'max_features': [None, 'auto'], 'loss':['ls', 'huber']
                    }
                ]
            },
            {
                'name': "Random Forest Regressor",
                'estimator':  RandomForestRegressor(),
                'params': [
                    {
                        'n_estimators': [30, 100, 120], 'max_depth': [None, 5, 15],
                        'max_features': [None, 'auto'],
                    },
                ]
            },
            {
                'name': "Ada Boost Regressor",
                'estimator': AdaBoostRegressor(),
                'params': [
                    {
                        'n_estimators': [20, 50, 100, 120]
                    }
                ]
            },
            {
                'name': "SVR",
                'estimator': SVR(),
                'params': [
                    {'degree': [2, 3, 4], 'C':[0.6, 1, 1.5]},
                ]
            },
            {
                'name': "Lasso",
                'estimator': Lasso(),
                'params': [
                    {'alpha': [0.5, 1, 2]},
                ]
            },
            {
                'name': "Linear Regression",
                'estimator': LinearRegression(),
                'params': [
                    {},
                ]
            },
        ]
    }
}


def getModels(isClassification: bool, computationLevel: str = 'medium'):
    modelType = "classification" if isClassification else "regression"

    return allModels[modelType][computationLevel]
