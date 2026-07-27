from math import log
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn._loss import HalfBinomialLoss, HalfMultinomialLoss
from sklearn.utils import Bunch, _safe_indexing, column_or_1d, get_tags, indexable

def _sigmoid_calibration(
    predictions, y, sample_weight=None, max_abs_prediction_threshold=30
):
    """Probability Calibration with sigmoid method (Platt 2000)

    Parameters
    ----------
    predictions : ndarray of shape (n_samples,)
        The decision function or predict proba for the samples.

    y : ndarray of shape (n_samples,)
        The targets.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted.

    Returns
    -------
    a : float
        The slope.

    b : float
        The intercept.

    References
    ----------
    Platt, "Probabilistic Outputs for Support Vector Machines"
    """
    predictions = column_or_1d(predictions)
    y = column_or_1d(y)

    F = predictions  # F follows Platt's notations

    scale_constant = 1.0
    max_prediction = np.max(np.abs(F))

    # If the predictions have large values we scale them in order to bring
    # them within a suitable range. This has no effect on the final
    # (prediction) result because linear models like Logisitic Regression
    # without a penalty are invariant to multiplying the features by a
    # constant.
    if max_prediction >= max_abs_prediction_threshold:
        scale_constant = max_prediction
        # We rescale the features in a copy: inplace rescaling could confuse
        # the caller and make the code harder to reason about.
        F = F / scale_constant

    # Bayesian priors (see Platt end of section 2.2):
    # It corresponds to the number of samples, taking into account the
    # `sample_weight`.
    mask_negative_samples = y <= 0
    if sample_weight is not None:
        prior0 = (sample_weight[mask_negative_samples]).sum()
        prior1 = (sample_weight[~mask_negative_samples]).sum()
    else:
        prior0 = float(np.sum(mask_negative_samples))
        prior1 = y.shape[0] - prior0
    T = np.zeros_like(y, dtype=predictions.dtype)
    T[y > 0] = (prior1 + 1.0) / (prior1 + 2.0)
    T[y <= 0] = 1.0 / (prior0 + 2.0)

    bin_loss = HalfBinomialLoss()

    def loss_grad(AB):
        # .astype below is needed to ensure y_true and raw_prediction have the
        # same dtype. With result = np.float64(0) * np.array([1, 2], dtype=np.float32)
        # - in Numpy 2, result.dtype is float64
        # - in Numpy<2, result.dtype is float32
        raw_prediction = -(AB[0] * F + AB[1]).astype(dtype=predictions.dtype)
        l, g = bin_loss.loss_gradient(
            y_true=T,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
        )
        loss = l.sum()
        # TODO: Remove casting to np.float64 when minimum supported SciPy is 1.11.2
        # With SciPy >= 1.11.2, the LBFGS implementation will cast to float64
        # https://github.com/scipy/scipy/pull/18825.
        # Here we cast to float64 to support SciPy < 1.11.2
        grad = np.asarray([-g @ F, -g.sum()], dtype=np.float64)
        return loss, grad

    AB0 = np.array([0.0, log((prior0 + 1.0) / (prior1 + 1.0))])

    opt_result = minimize(
        loss_grad,
        AB0,
        method="L-BFGS-B",
        jac=True,
        options={
            "gtol": 1e-6,
            "ftol": 64 * np.finfo(float).eps,
        },
    )
    AB_ = opt_result.x

    # The tuned multiplicative parameter is converted back to the original
    # input feature scale. The offset parameter does not need rescaling since
    # we did not rescale the outcome variable.
    return AB_[0] / scale_constant, AB_[1]
