import warnings
from sklearn.utils._array_api import (
    _convert_to_numpy,
    device,
    get_namespace,
    get_namespace_and_device,
    move_to,
)
from sklearn.utils.validation import _check_method_params, _num_samples

def _enforce_prediction_order(classes, predictions, n_classes, method):
    """Ensure that prediction arrays have correct column order

    When doing cross-validation, if one or more classes are
    not present in the subset of data used for training,
    then the output prediction array might not have the same
    columns as other folds. Use the list of class names
    (assumed to be ints) to enforce the correct column order.

    Note that `classes` is the list of classes in this fold
    (a subset of the classes in the full training set)
    and `n_classes` is the number of classes in the full training set.
    """
    xp, _ = get_namespace(predictions, classes)
    classes_length = classes.shape[0]
    if n_classes != classes_length:
        recommendation = (
            "To fix this, use a cross-validation "
            "technique resulting in properly "
            "stratified folds"
        )
        warnings.warn(
            "Number of classes in training fold ({}) does "
            "not match total number of classes ({}). "
            "Results may not be appropriate for your use case. "
            "{}".format(classes_length, n_classes, recommendation),
            RuntimeWarning,
        )
        if method == "decision_function":
            if predictions.ndim == 2 and predictions.shape[1] != classes_length:
                # This handles the case when the shape of predictions
                # does not match the number of classes used to train
                # it with. This case is found when sklearn.svm.SVC is
                # set to `decision_function_shape='ovo'`.
                raise ValueError(
                    "Output shape {} of {} does not match "
                    "number of classes ({}) in fold. "
                    "Irregular decision_function outputs "
                    "are not currently supported by "
                    "cross_val_predict".format(
                        predictions.shape, method, classes_length
                    )
                )
            if classes_length <= 2:
                # In this special case, `predictions` contains a 1D array.
                raise ValueError(
                    "Only {} class/es in training fold, but {} "
                    "in overall dataset. This "
                    "is not supported for decision_function "
                    "with imbalanced folds. {}".format(
                        classes_length, n_classes, recommendation
                    )
                )

        float_min = xp.finfo(predictions.dtype).min
        default_values = {
            "decision_function": float_min,
            "predict_log_proba": float_min,
            "predict_proba": 0,
        }
        predictions_for_all_classes = xp.full(
            (_num_samples(predictions), n_classes),
            default_values[method],
            dtype=predictions.dtype,
        )
        predictions_for_all_classes[:, classes] = predictions
        predictions = predictions_for_all_classes
    return predictions
