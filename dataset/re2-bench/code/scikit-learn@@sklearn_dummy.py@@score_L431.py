from numbers import Integral, Real
import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
    _fit_context,
)
from sklearn.utils._param_validation import Interval, StrOptions

class DummyClassifier(MultiOutputMixin, ClassifierMixin, BaseEstimator):
    """DummyClassifier makes predictions that ignore the input features.

    This classifier serves as a simple baseline to compare against other more
    complex classifiers.

    The specific behavior of the baseline is selected with the `strategy`
    parameter.

    All strategies make predictions that ignore the input feature values passed
    as the `X` argument to `fit` and `predict`. The predictions, however,
    typically depend on values observed in the `y` parameter passed to `fit`.

    Note that the "stratified" and "uniform" strategies lead to
    non-deterministic predictions that can be rendered deterministic by setting
    the `random_state` parameter if needed. The other strategies are naturally
    deterministic and, once fit, always return the same constant prediction
    for any value of `X`.

    Read more in the :ref:`User Guide <dummy_estimators>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    strategy : {"most_frequent", "prior", "stratified", "uniform",             "constant"}, default="prior"
        Strategy to use to generate predictions.

        * "most_frequent": the `predict` method always returns the most
          frequent class label in the observed `y` argument passed to `fit`.
          The `predict_proba` method returns the matching one-hot encoded
          vector.
        * "prior": the `predict` method always returns the most frequent
          class label in the observed `y` argument passed to `fit` (like
          "most_frequent"). ``predict_proba`` always returns the empirical
          class distribution of `y` also known as the empirical class prior
          distribution.
        * "stratified": the `predict_proba` method randomly samples one-hot
          vectors from a multinomial distribution parametrized by the empirical
          class prior probabilities.
          The `predict` method returns the class label which got probability
          one in the one-hot vector of `predict_proba`.
          Each sampled row of both methods is therefore independent and
          identically distributed.
        * "uniform": generates predictions uniformly at random from the list
          of unique classes observed in `y`, i.e. each class has equal
          probability.
        * "constant": always predicts a constant label that is provided by
          the user. This is useful for metrics that evaluate a non-majority
          class.

          .. versionchanged:: 0.24
             The default value of `strategy` has changed to "prior" in version
             0.24.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness to generate the predictions when
        ``strategy='stratified'`` or ``strategy='uniform'``.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    constant : int or str or array-like of shape (n_outputs,), default=None
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of such arrays
        Unique class labels observed in `y`. For multi-output classification
        problems, this attribute is a list of arrays as each output has an
        independent set of possible classes.

    n_classes_ : int or list of int
        Number of label for each output.

    class_prior_ : ndarray of shape (n_classes,) or list of such arrays
        Frequency of each class observed in `y`. For multioutput classification
        problems, this is computed independently for each output.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X` has
        feature names that are all strings.

    n_outputs_ : int
        Number of outputs.

    sparse_output_ : bool
        True if the array returned from predict is to be in sparse CSC format.
        Is automatically set to True if the input `y` is passed in sparse
        format.

    See Also
    --------
    DummyRegressor : Regressor that makes predictions using simple rules.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.dummy import DummyClassifier
    >>> X = np.array([-1, 1, 1, 1])
    >>> y = np.array([0, 1, 1, 1])
    >>> dummy_clf = DummyClassifier(strategy="most_frequent")
    >>> dummy_clf.fit(X, y)
    DummyClassifier(strategy='most_frequent')
    >>> dummy_clf.predict(X)
    array([1, 1, 1, 1])
    >>> dummy_clf.score(X, y)
    0.75
    """
    _parameter_constraints: dict = {'strategy': [StrOptions({'most_frequent', 'prior', 'stratified', 'uniform', 'constant'})], 'random_state': ['random_state'], 'constant': [Integral, str, 'array-like', None]}

    def __init__(self, *, strategy='prior', random_state=None, constant=None):
        self.strategy = strategy
        self.random_state = random_state
        self.constant = constant

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : None or array-like of shape (n_samples, n_features)
            Test samples. Passing None as test samples gives the same result
            as passing real test samples, since DummyClassifier
            operates independently of the sampled observations.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) w.r.t. y.
        """
        if X is None:
            X = np.zeros(shape=(len(y), 1))
        return super().score(X, y, sample_weight)
