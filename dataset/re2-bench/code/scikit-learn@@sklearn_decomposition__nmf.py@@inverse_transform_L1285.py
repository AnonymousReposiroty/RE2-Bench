from abc import ABC
from numbers import Integral, Real
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.validation import check_is_fitted, check_non_negative, validate_data

class _BaseNMF(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, ABC):
    """Base class for NMF and MiniBatchNMF."""
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 1, None, closed='left'), None, StrOptions({'auto'})], 'init': [StrOptions({'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}), None], 'beta_loss': [StrOptions({'frobenius', 'kullback-leibler', 'itakura-saito'}), Real], 'tol': [Interval(Real, 0, None, closed='left')], 'max_iter': [Interval(Integral, 1, None, closed='left')], 'random_state': ['random_state'], 'alpha_W': [Interval(Real, 0, None, closed='left')], 'alpha_H': [Interval(Real, 0, None, closed='left'), StrOptions({'same'})], 'l1_ratio': [Interval(Real, 0, 1, closed='both')], 'verbose': ['verbose']}

    def __init__(self, n_components='auto', *, init=None, beta_loss='frobenius', tol=0.0001, max_iter=200, random_state=None, alpha_W=0.0, alpha_H='same', l1_ratio=0.0, verbose=0):
        self.n_components = n_components
        self.init = init
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
        self.verbose = verbose

    def inverse_transform(self, X):
        """Transform data back to its original space.

        .. versionadded:: 0.18

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_components)
            Transformed data matrix.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Returns a data matrix of the original shape.
        """
        check_is_fitted(self)
        return X @ self.components_
