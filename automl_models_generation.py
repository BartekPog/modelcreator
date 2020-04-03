from sklearn.model_selection import GridSearchCV
from automl_models import classificationModels, regressionModels


def generateModel(X, y, isClassification, verbose=True):
    if isClassification:
        models = classificationModels
        scoring = 'accuracy'
    else:
        models = regressionModels
        scoring = 'neg_root_mean_squared_error'

    finalModel = models[0]

    for model in models:
        if verbose:
            print("Testing: ", model['name'], end='')

        model['grid_search_result'] = GridSearchCV(model['estimator'], model['params'],
                                                   cv=5, scoring=scoring, n_jobs=-1)
        model['grid_search_result'].fit(X, y)

        if verbose:
            print(" ", model['grid_search_result'].best_score_)

        if model['grid_search_result'].best_score_ > finalModel['grid_search_result'].best_score_:
            finalModel = model

    return {
        'estimator': finalModel['grid_search_result'].best_estimator_,
        'name': finalModel['name'],
        'params': finalModel['grid_search_result'].best_params_
    }
