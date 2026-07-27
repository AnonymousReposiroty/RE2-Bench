from abc import ABCMeta, abstractmethod
from sklearn.base import clone

class Kernel(metaclass=ABCMeta):
    """Base class for all kernels.

    .. versionadded:: 0.18

    Examples
    --------
    >>> from sklearn.gaussian_process.kernels import Kernel, RBF
    >>> import numpy as np
    >>> class CustomKernel(Kernel):
    ...     def __init__(self, length_scale=1.0):
    ...         self.length_scale = length_scale
    ...     def __call__(self, X, Y=None):
    ...         if Y is None:
    ...             Y = X
    ...         return np.inner(X, X if Y is None else Y) ** 2
    ...     def diag(self, X):
    ...         return np.ones(X.shape[0])
    ...     def is_stationary(self):
    ...         return True
    >>> kernel = CustomKernel(length_scale=2.0)
    >>> X = np.array([[1, 2], [3, 4]])
    >>> print(kernel(X))
    [[ 25 121]
     [121 625]]
    """

    def clone_with_theta(self, theta):
        """Returns a clone of self with given hyperparameters theta.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The hyperparameters
        """
        cloned = clone(self)
        cloned.theta = theta
        return cloned
