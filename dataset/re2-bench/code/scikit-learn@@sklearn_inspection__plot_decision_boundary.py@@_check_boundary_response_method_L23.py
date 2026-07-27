from sklearn.base import is_regressor
from sklearn.utils.validation import (
    _is_arraylike_not_scalar,
    _is_pandas_df,
    _is_polars_df,
    _num_features,
    check_is_fitted,
)

def _check_boundary_response_method(estimator, response_method, class_of_interest):
    """Validate the response methods to be used with the fitted estimator.

    Parameters
    ----------
    estimator : object
        Fitted estimator to check.

    response_method : {'auto', 'decision_function', 'predict_proba', 'predict'}
        Specifies whether to use :term:`decision_function`, :term:`predict_proba`,
        :term:`predict` as the target response. If set to 'auto', the response method is
        tried in the before mentioned order.

    class_of_interest : int, float, bool, str or None
        The class considered when plotting the decision. Cannot be None if
        multiclass and `response_method` is 'predict_proba' or 'decision_function'.

        .. versionadded:: 1.4

    Returns
    -------
    prediction_method : list of str or str
        The name or list of names of the response methods to use.
    """
    has_classes = hasattr(estimator, "classes_")
    if has_classes and _is_arraylike_not_scalar(estimator.classes_[0]):
        msg = "Multi-label and multi-output multi-class classifiers are not supported"
        raise ValueError(msg)

    if response_method == "auto":
        if is_regressor(estimator):
            prediction_method = "predict"
        else:
            prediction_method = ["decision_function", "predict_proba", "predict"]
    else:
        prediction_method = response_method

    return prediction_method
