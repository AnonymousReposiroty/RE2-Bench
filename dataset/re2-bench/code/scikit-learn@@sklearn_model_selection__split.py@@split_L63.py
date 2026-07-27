import warnings

class _UnsupportedGroupCVMixin:
    """Mixin for splitters that do not support Groups."""

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Always ignored, exists for API compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        if groups is not None:
            warnings.warn(f'The groups parameter is ignored by {self.__class__.__name__}', UserWarning)
        return super().split(X, y, groups=groups)
