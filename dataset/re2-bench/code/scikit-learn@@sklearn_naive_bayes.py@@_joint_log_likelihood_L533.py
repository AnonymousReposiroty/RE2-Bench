from numbers import Integral, Real
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

    def _joint_log_likelihood(self, X):
        xp, _ = get_namespace(X)
        joint_log_likelihood = []
        for i in range(size(self.classes_)):
            jointi = xp.log(self.class_prior_[i])
            n_ij = -0.5 * xp.sum(xp.log(2.0 * xp.pi * self.var_[i, :]))
            n_ij = n_ij - 0.5 * xp.sum((X - self.theta_[i, :]) ** 2 / self.var_[i, :], axis=1)
            joint_log_likelihood.append(jointi + n_ij)
        joint_log_likelihood = xp.stack(joint_log_likelihood).T
        return joint_log_likelihood
