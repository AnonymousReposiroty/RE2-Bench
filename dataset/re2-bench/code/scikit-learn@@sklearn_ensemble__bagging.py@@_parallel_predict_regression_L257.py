def _parallel_predict_regression(estimators, estimators_features, X, params):
    """Private function used to compute predictions within a job."""
    return sum(
        estimator.predict(X[:, features], **params)
        for estimator, features in zip(estimators, estimators_features)
    )
