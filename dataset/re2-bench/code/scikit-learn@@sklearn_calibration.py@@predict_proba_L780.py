import numpy as np
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils._array_api import (
    _convert_to_numpy,
    _half_multinomial_loss,
    _is_numpy_namespace,
    get_namespace,
    get_namespace_and_device,
    move_to,
)
from sklearn.utils._response import _get_response_values, _process_predict_proba
from sklearn.utils.validation import (
    _check_method_params,
    _check_pos_label_consistency,
    _check_response_method,
    _check_sample_weight,
    _num_samples,
    check_array,
    check_consistent_length,
    check_is_fitted,
)

class _CalibratedClassifier:
    """Pipeline-like chaining a fitted classifier and its fitted calibrators.

    Parameters
    ----------
    estimator : estimator instance
        Fitted classifier.

    calibrators : list of fitted estimator instances
        List of fitted calibrators (either 'IsotonicRegression' or
        '_SigmoidCalibration'). The number of calibrators equals the number of
        classes. However, if there are 2 classes, the list contains only one
        fitted calibrator.

    classes : array-like of shape (n_classes,)
        All the prediction classes.

    method : {'sigmoid', 'isotonic'}, default='sigmoid'
        The method to use for calibration. Can be 'sigmoid' which
        corresponds to Platt's method or 'isotonic' which is a
        non-parametric approach based on isotonic regression.
    """

    def __init__(self, estimator, calibrators, *, classes, method='sigmoid'):
        self.estimator = estimator
        self.calibrators = calibrators
        self.classes = classes
        self.method = method

    def predict_proba(self, X):
        """Calculate calibrated probabilities.

        Calculates classification calibrated probabilities
        for each class, in a one-vs-all manner, for `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The sample data.

        Returns
        -------
        proba : array, shape (n_samples, n_classes)
            The predicted probabilities. Can be exact zeros.
        """
        predictions, _ = _get_response_values(self.estimator, X, response_method=['decision_function', 'predict_proba'])
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        n_classes = self.classes.shape[0]
        proba = np.zeros((_num_samples(X), n_classes))
        if self.method in ('sigmoid', 'isotonic'):
            label_encoder = LabelEncoder().fit(self.classes)
            pos_class_indices = label_encoder.transform(self.estimator.classes_)
            for class_idx, this_pred, calibrator in zip(pos_class_indices, predictions.T, self.calibrators):
                if n_classes == 2:
                    class_idx += 1
                proba[:, class_idx] = calibrator.predict(this_pred)
            if n_classes == 2:
                proba[:, 0] = 1.0 - proba[:, 1]
            else:
                denominator = np.sum(proba, axis=1)[:, np.newaxis]
                uniform_proba = np.full_like(proba, 1 / n_classes)
                proba = np.divide(proba, denominator, out=uniform_proba, where=denominator != 0)
        elif self.method == 'temperature':
            xp, _ = get_namespace(predictions)
            if n_classes == 2 and predictions.shape[-1] == 1:
                response_method_name = _check_response_method(self.estimator, ['decision_function', 'predict_proba']).__name__
                if response_method_name == 'predict_proba':
                    predictions = xp.concat([1 - predictions, predictions], axis=1)
            proba = self.calibrators[0].predict(predictions)
        proba[(1.0 < proba) & (proba <= 1.0 + 1e-05)] = 1.0
        return proba
