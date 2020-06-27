from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from .models import classificationModels, regressionModels


def generateModel(X, y, isClassification, metrics, verbose=True):
    if (metrics == None):
        scoring = 'accuracy' if isClassification else 'neg_root_mean_squared_error'
    else:
        scoring = metrics if isinstance(metrics, str) else make_scorer(metrics)

    models = classificationModels if isClassification else regressionModels
    finalModel = models[0]

    for model in models:
        if verbose:
            print("Testing: ", model['name'], end='')

        model['grid_search_result'] = GridSearchCV(model['estimator'], model['params'],
                                                   cv=5, scoring=scoring, n_jobs=-1)
        model['grid_search_result'].fit(X, y)

        if verbose:
            print(" {0:.4f}".format(model['grid_search_result'].best_score_))

        if model['grid_search_result'].best_score_ > finalModel['grid_search_result'].best_score_:
            finalModel = model

    if verbose:
        print("Chosen model: ", finalModel['name'],
              "{0:.4f}".format(finalModel['grid_search_result'].best_score_))

    return {
        'estimator': finalModel['grid_search_result'].best_estimator_,
        'name': finalModel['name'],
        'params': finalModel['grid_search_result'].best_params_,
    }
