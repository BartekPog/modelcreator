import dask_ml.model_selection as dcv
from dask.diagnostics import ProgressBar
from sklearn.metrics import make_scorer

from .models import getModels


def generateModel(X, y, isClassification: bool, metrics, verbose: bool = True, cv: int = 3, computationLevel: str = 'medium'):

    if (metrics == None):
        scoring = 'accuracy' if isClassification else 'neg_root_mean_squared_error'
    else:
        scoring = metrics if isinstance(metrics, str) else make_scorer(metrics)

    models = getModels(isClassification=isClassification,
                       computationLevel=computationLevel)

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

        print("\nParams:")
        params = finalModel['grid_search_result'].best_params_

        if(len(params) > 0):
            for key, value in params.items():
                print("\t{}: {}".format(key, value))

        else:
            print("\tdefault")

        print("")

    return {
        'estimator': finalModel['grid_search_result'].best_estimator_,
        'name': finalModel['name'],
        'params': finalModel['grid_search_result'].best_params_,
    }
