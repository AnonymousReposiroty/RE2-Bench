from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

class FrozenEstimator(BaseEstimator):
    """Estimator that wraps a fitted estimator to prevent re-fitting.

    This meta-estimator takes an estimator and freezes it, in the sense that calling
    `fit` on it has no effect. `fit_predict` and `fit_transform` are also disabled.
    All other methods are delegated to the original estimator and original estimator's
    attributes are accessible as well.

    This is particularly useful when you have a fitted or a pre-trained model as a
    transformer in a pipeline, and you'd like `pipeline.fit` to have no effect on this
    step.

    Parameters
    ----------
    estimator : estimator
        The estimator which is to be kept frozen.

    See Also
    --------
    None: No similar entry in the scikit-learn documentation.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.frozen import FrozenEstimator
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(random_state=0)
    >>> clf = LogisticRegression(random_state=0).fit(X, y)
    >>> frozen_clf = FrozenEstimator(clf)
    >>> frozen_clf.fit(X, y)  # No-op
    FrozenEstimator(estimator=LogisticRegression(random_state=0))
    >>> frozen_clf.predict(X)  # Predictions from `clf.predict`
    array(...)
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, *args, **kwargs):
        """No-op.

        As a frozen estimator, calling `fit` has no effect.

        Parameters
        ----------
        X : object
            Ignored.

        y : object
            Ignored.

        *args : tuple
            Additional positional arguments. Ignored, but present for API compatibility
            with `self.estimator`.

        **kwargs : dict
            Additional keyword arguments. Ignored, but present for API compatibility
            with `self.estimator`.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        check_is_fitted(self.estimator)
        return self
