from numbers import Integral, Real
import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
    _fit_context,
)
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import (
    _check_sample_weight,
    _num_samples,
    check_array,
    check_consistent_length,
    check_is_fitted,
    validate_data,
)

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

    def predict_proba(self, X):
        """
        Return probability estimates for the test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        P : ndarray of shape (n_samples, n_classes) or list of such arrays
            Returns the probability of the sample for each class in
            the model, where classes are ordered arithmetically, for each
            output.
        """
        check_is_fitted(self)
        n_samples = _num_samples(X)
        rs = check_random_state(self.random_state)
        n_classes_ = self.n_classes_
        classes_ = self.classes_
        class_prior_ = self.class_prior_
        constant = self.constant
        if self.n_outputs_ == 1:
            n_classes_ = [n_classes_]
            classes_ = [classes_]
            class_prior_ = [class_prior_]
            constant = [constant]
        P = []
        for k in range(self.n_outputs_):
            if self._strategy == 'most_frequent':
                ind = class_prior_[k].argmax()
                out = np.zeros((n_samples, n_classes_[k]), dtype=np.float64)
                out[:, ind] = 1.0
            elif self._strategy == 'prior':
                out = np.ones((n_samples, 1)) * class_prior_[k]
            elif self._strategy == 'stratified':
                out = rs.multinomial(1, class_prior_[k], size=n_samples)
                out = out.astype(np.float64)
            elif self._strategy == 'uniform':
                out = np.ones((n_samples, n_classes_[k]), dtype=np.float64)
                out /= n_classes_[k]
            elif self._strategy == 'constant':
                ind = np.where(classes_[k] == constant[k])
                out = np.zeros((n_samples, n_classes_[k]), dtype=np.float64)
                out[:, ind] = 1.0
            P.append(out)
        if self.n_outputs_ == 1:
            P = P[0]
        return P
