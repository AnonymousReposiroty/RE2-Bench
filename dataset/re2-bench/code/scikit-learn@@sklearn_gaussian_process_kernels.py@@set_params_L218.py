from abc import ABCMeta, abstractmethod
from inspect import signature

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

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = dict()
        cls = self.__class__
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        init_sign = signature(init)
        args, varargs = ([], [])
        for parameter in init_sign.parameters.values():
            if parameter.kind != parameter.VAR_KEYWORD and parameter.name != 'self':
                args.append(parameter.name)
            if parameter.kind == parameter.VAR_POSITIONAL:
                varargs.append(parameter.name)
        if len(varargs) != 0:
            raise RuntimeError("scikit-learn kernels should always specify their parameters in the signature of their __init__ (no varargs). %s doesn't follow this convention." % (cls,))
        for arg in args:
            params[arg] = getattr(self, arg)
        return params

    def set_params(self, **params):
        """Set the parameters of this kernel.

        The method works on simple kernels as well as on nested kernels.
        The latter have parameters of the form ``<component>__<parameter>``
        so that it's possible to update each component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            split = key.split('__', 1)
            if len(split) > 1:
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for kernel %s. Check the list of available parameters with `kernel.get_params().keys()`.' % (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for kernel %s. Check the list of available parameters with `kernel.get_params().keys()`.' % (key, self.__class__.__name__))
                setattr(self, key, value)
        return self
