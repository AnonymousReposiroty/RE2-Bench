from sklearn.model_selection._validation import _score
from sklearn.utils.metaestimators import _safe_split, available_if
from sklearn.utils.validation import (
    _check_method_params,
    _estimator_has,
    check_is_fitted,
    validate_data,
)

def _rfe_single_fit(rfe, estimator, X, y, train, test, scorer, routed_params):
    """
    Return the score and n_features per step for a fit across one fold.
    """
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    fit_params = _check_method_params(
        X, params=routed_params.estimator.fit, indices=train
    )
    score_params = _check_method_params(
        X=X, params=routed_params.scorer.score, indices=test
    )

    rfe._fit(
        X_train,
        y_train,
        lambda estimator, features: _score(
            estimator,
            X_test[:, features],
            y_test,
            scorer,
            score_params=score_params,
        ),
        **fit_params,
    )

    return rfe.step_scores_, rfe.step_support_, rfe.step_ranking_, rfe.step_n_features_
