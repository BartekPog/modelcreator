from sklearn.metrics import make_scorer
from .models import classificationModels, regressionModels

# from sklearn.model_selection import GridSearchCV
import dask_ml.model_selection as dcv
from dask.diagnostics import ProgressBar


def generateModel(X, y, isClassification, metrics, verbose=True, cv=3):
    if (metrics == None):
        scoring = 'accuracy' if isClassification else 'neg_root_mean_squared_error'
    else:
        scoring = metrics if isinstance(metrics, str) else make_scorer(metrics)

    models = classificationModels if isClassification else regressionModels
    finalModel = models[0]

    for model in models:

        model['grid_search_result'] = dcv.GridSearchCV(model['estimator'], param_grid=model['params'],
                                                       cv=cv, scoring=scoring)

        if verbose:
            print("Testing: ", model['name'])

            with ProgressBar(minimum=1):
                model['grid_search_result'].fit(X, y)

            print("Score: {0:.4f}\n".format(
                model['grid_search_result'].best_score_))

        else:
            model['grid_search_result'].fit(X, y)

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
