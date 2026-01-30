import numbers
from abc import ABCMeta, abstractmethod

class _BaseKFold(BaseCrossValidator, metaclass=ABCMeta):
    """Base class for K-Fold cross-validators and TimeSeriesSplit."""

    @abstractmethod
    def __init__(self, n_splits, *, shuffle, random_state):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError('The number of folds must be of Integral type. %s of type %s was passed.' % (n_splits, type(n_splits)))
        n_splits = int(n_splits)
        if n_splits <= 1:
            raise ValueError('k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits={0}.'.format(n_splits))
        if not isinstance(shuffle, bool):
            raise TypeError('shuffle must be True or False; got {0}'.format(shuffle))
        if not shuffle and random_state is not None:
            raise ValueError('Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.')
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations as set with the `n_splits` param
        when instantiating the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Always ignored, exists for API compatibility.

        y : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        groups : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits
