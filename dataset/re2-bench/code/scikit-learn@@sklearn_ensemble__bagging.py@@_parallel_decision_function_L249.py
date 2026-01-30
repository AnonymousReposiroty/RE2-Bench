def _parallel_decision_function(estimators, estimators_features, X, params):
    """Private function used to compute decisions within a job."""
    return sum(
        estimator.decision_function(X[:, features], **params)
        for estimator, features in zip(estimators, estimators_features)
    )
