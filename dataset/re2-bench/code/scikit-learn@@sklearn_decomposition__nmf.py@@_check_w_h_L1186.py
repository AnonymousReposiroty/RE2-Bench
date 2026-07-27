import warnings
from abc import ABC
from numbers import Integral, Real
import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils._param_validation import Interval, StrOptions, validate_params

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

    def _check_w_h(self, X, W, H, update_H):
        """Check W and H, or initialize them."""
        n_samples, n_features = X.shape
        if self.init == 'custom' and update_H:
            _check_init(H, (self._n_components, n_features), 'NMF (input H)')
            _check_init(W, (n_samples, self._n_components), 'NMF (input W)')
            if self._n_components == 'auto':
                self._n_components = H.shape[0]
            if H.dtype != X.dtype or W.dtype != X.dtype:
                raise TypeError('H and W should have the same dtype as X. Got H.dtype = {} and W.dtype = {}.'.format(H.dtype, W.dtype))
        elif not update_H:
            if W is not None:
                warnings.warn('When update_H=False, the provided initial W is not used.', RuntimeWarning)
            _check_init(H, (self._n_components, n_features), 'NMF (input H)')
            if self._n_components == 'auto':
                self._n_components = H.shape[0]
            if H.dtype != X.dtype:
                raise TypeError('H should have the same dtype as X. Got H.dtype = {}.'.format(H.dtype))
            if self.solver == 'mu':
                avg = np.sqrt(X.mean() / self._n_components)
                W = np.full((n_samples, self._n_components), avg, dtype=X.dtype)
            else:
                W = np.zeros((n_samples, self._n_components), dtype=X.dtype)
        else:
            if W is not None or H is not None:
                warnings.warn("When init!='custom', provided W or H are ignored. Set  init='custom' to use them as initialization.", RuntimeWarning)
            if self._n_components == 'auto':
                self._n_components = X.shape[1]
            W, H = _initialize_nmf(X, self._n_components, init=self.init, random_state=self.random_state)
        return (W, H)
