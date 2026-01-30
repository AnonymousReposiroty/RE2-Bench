from abc import ABC
from numbers import Integral, Real
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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ['float64', 'float32']
        return tags
