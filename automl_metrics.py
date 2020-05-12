from typing import Union
from sklearn.metrics import make_scorer

classificationLibraryMetrics = set([
    'accuracy',
    'balanced_accuracy',
    'average_precision',
    'neg_brier_score',
    'f1',
    'f1_micro',
    'f1_macro',
    'f1_weighted',
    'f1_samples',
    'neg_log_loss',
    'precision',
    'precision_micro',
    'precision_macro',
    'precision_weighted',
    'precision_samples',
    'recall',
    'recall_micro',
    'recall_macro',
    'recall_weighted',
    'recall_samples',
    'jaccard',
    'jaccard_micro',
    'jaccard_macro',
    'jaccard_weighted',
    'jaccard_samples',
    'roc_auc',
    'roc_auc_ovr',
    'roc_auc_ovo',
    'roc_auc_ovr_weighted',
    'roc_auc_ovo_weighted'
])

regressionLibraryMetrics = set([
    'explained_variance',
    'max_error',
    'neg_mean_absolute_error',
    'neg_mean_squared_error',
    'neg_root_mean_squared_error',
    'neg_mean_squared_log_error',
    'neg_median_absolute_error',
    'r2',
    'neg_mean_poisson_deviance',
    'neg_mean_gamma_deviance',
])


def isValidMetrics(metrics: Union[None, str, callable], isClassifier: bool) -> bool:
    if(metrics == None):
        return True

    if(isinstance(metrics, str)):
        possibleMetrics = classificationLibraryMetrics if isClassifier else regressionLibraryMetrics

        return metrics in possibleMetrics

    try:
        _ = make_scorer(metrics)
        return True
    except:
        return False

    return False
