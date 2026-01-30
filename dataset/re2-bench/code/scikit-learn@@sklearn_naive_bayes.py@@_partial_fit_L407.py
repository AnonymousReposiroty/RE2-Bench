from numbers import Integral, Real
import numpy as np
import sklearn.externals.array_api_extra as xpx
from sklearn.utils._array_api import (
    _average,
    _convert_to_numpy,
    _find_matching_floating_dtype,
    _isin,
    _logsumexp,
    get_namespace,
    get_namespace_and_device,
    size,
)
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.utils.validation import (
    _check_n_features,
    _check_sample_weight,
    check_is_fitted,
    check_non_negative,
    validate_data,
)

class GaussianNB(_BaseNB):
    """
    Gaussian Naive Bayes (GaussianNB).

    Can perform online updates to model parameters via :meth:`partial_fit`.
    For details on algorithm used to update feature means and variance online,
    see `Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque
    <http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf>`_.

    Read more in the :ref:`User Guide <gaussian_naive_bayes>`.

    Parameters
    ----------
    priors : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability.

        .. versionadded:: 0.20

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        number of training samples observed in each class.

    class_prior_ : ndarray of shape (n_classes,)
        probability of each class.

    classes_ : ndarray of shape (n_classes,)
        class labels known to the classifier.

    epsilon_ : float
        absolute additive value to variances.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    var_ : ndarray of shape (n_classes, n_features)
        Variance of each feature per class.

        .. versionadded:: 1.0

    theta_ : ndarray of shape (n_classes, n_features)
        mean of each feature per class.

    See Also
    --------
    BernoulliNB : Naive Bayes classifier for multivariate Bernoulli models.
    CategoricalNB : Naive Bayes classifier for categorical features.
    ComplementNB : Complement Naive Bayes classifier.
    MultinomialNB : Naive Bayes classifier for multinomial models.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from sklearn.naive_bayes import GaussianNB
    >>> clf = GaussianNB()
    >>> clf.fit(X, Y)
    GaussianNB()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    >>> clf_pf = GaussianNB()
    >>> clf_pf.partial_fit(X, Y, np.unique(Y))
    GaussianNB()
    >>> print(clf_pf.predict([[-0.8, -1]]))
    [1]
    """
    _parameter_constraints: dict = {'priors': ['array-like', None], 'var_smoothing': [Interval(Real, 0, None, closed='left')]}

    def __init__(self, *, priors=None, var_smoothing=1e-09):
        self.priors = priors
        self.var_smoothing = var_smoothing

    @staticmethod
    def _update_mean_variance(n_past, mu, var, X, sample_weight=None):
        """Compute online update of Gaussian mean and variance.

        Given starting sample count, mean, and variance, a new set of
        points X, and optionally sample weights, return the updated mean and
        variance. (NB - each dimension (column) in X is treated as independent
        -- you get variance, not covariance).

        Can take scalar mean and variance, or vector mean and variance to
        simultaneously update a number of independent Gaussians.

        See Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

        Parameters
        ----------
        n_past : int
            Number of samples represented in old mean and variance. If sample
            weights were given, this should contain the sum of sample
            weights represented in old mean and variance.

        mu : array-like of shape (number of Gaussians,)
            Means for Gaussians in original set.

        var : array-like of shape (number of Gaussians,)
            Variances for Gaussians in original set.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        total_mu : array-like of shape (number of Gaussians,)
            Updated mean for each Gaussian over the combined set.

        total_var : array-like of shape (number of Gaussians,)
            Updated variance for each Gaussian over the combined set.
        """
        xp, _ = get_namespace(X)
        if X.shape[0] == 0:
            return (mu, var)
        if sample_weight is not None:
            n_new = float(xp.sum(sample_weight))
            if np.isclose(n_new, 0.0):
                return (mu, var)
            new_mu = _average(X, axis=0, weights=sample_weight, xp=xp)
            new_var = _average((X - new_mu) ** 2, axis=0, weights=sample_weight, xp=xp)
        else:
            n_new = X.shape[0]
            new_var = xp.var(X, axis=0)
            new_mu = xp.mean(X, axis=0)
        if n_past == 0:
            return (new_mu, new_var)
        n_total = float(n_past + n_new)
        total_mu = (n_new * new_mu + n_past * mu) / n_total
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = old_ssd + new_ssd + n_new * n_past / n_total * (mu - new_mu) ** 2
        total_var = total_ssd / n_total
        return (total_mu, total_var)

    def _partial_fit(self, X, y, classes=None, _refit=False, sample_weight=None):
        """Actual implementation of Gaussian NB fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        _refit : bool, default=False
            If true, act as though this were the first time we called
            _partial_fit (ie, throw away any past fitting and start over).

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
        """
        if _refit:
            self.classes_ = None
        first_call = _check_partial_fit_first_call(self, classes)
        X, y = validate_data(self, X, y, reset=first_call)
        xp, _, device_ = get_namespace_and_device(X)
        float_dtype = _find_matching_floating_dtype(X, xp=xp)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=float_dtype)
        xp_y, _ = get_namespace(y)
        self.epsilon_ = self.var_smoothing * xp.max(xp.var(X, axis=0))
        if first_call:
            n_features = X.shape[1]
            n_classes = self.classes_.shape[0]
            self.theta_ = xp.zeros((n_classes, n_features), dtype=float_dtype, device=device_)
            self.var_ = xp.zeros((n_classes, n_features), dtype=float_dtype, device=device_)
            self.class_count_ = xp.zeros(n_classes, dtype=float_dtype, device=device_)
            if self.priors is not None:
                priors = xp.asarray(self.priors, dtype=float_dtype, device=device_)
                if priors.shape[0] != n_classes:
                    raise ValueError('Number of priors must match number of classes.')
                if not xpx.isclose(xp.sum(priors), 1.0):
                    raise ValueError('The sum of the priors should be 1.')
                if xp.any(priors < 0):
                    raise ValueError('Priors must be non-negative.')
                self.class_prior_ = priors
            else:
                self.class_prior_ = xp.zeros(self.classes_.shape[0], dtype=float_dtype, device=device_)
        else:
            if X.shape[1] != self.theta_.shape[1]:
                msg = 'Number of features %d does not match previous data %d.'
                raise ValueError(msg % (X.shape[1], self.theta_.shape[1]))
            self.var_[:, :] -= self.epsilon_
        classes = self.classes_
        unique_y = xp_y.unique_values(y)
        unique_y_in_classes = _isin(unique_y, classes, xp=xp_y)
        if not xp_y.all(unique_y_in_classes):
            raise ValueError('The target label(s) %s in y do not exist in the initial classes %s' % (unique_y[~unique_y_in_classes], classes))
        for y_i in unique_y:
            i = int(xp_y.searchsorted(classes, y_i))
            y_i_mask = xp.asarray(y == y_i, device=device_)
            X_i = X[y_i_mask]
            if sample_weight is not None:
                sw_i = sample_weight[y_i_mask]
                N_i = xp.sum(sw_i)
            else:
                sw_i = None
                N_i = X_i.shape[0]
            new_theta, new_sigma = self._update_mean_variance(self.class_count_[i], self.theta_[i, :], self.var_[i, :], X_i, sw_i)
            self.theta_[i, :] = new_theta
            self.var_[i, :] = new_sigma
            self.class_count_[i] += N_i
        self.var_[:, :] += self.epsilon_
        if self.priors is None:
            self.class_prior_ = self.class_count_ / xp.sum(self.class_count_)
        return self
