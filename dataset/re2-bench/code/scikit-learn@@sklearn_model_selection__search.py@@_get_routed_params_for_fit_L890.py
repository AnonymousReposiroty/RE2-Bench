import numbers
import warnings
from abc import ABCMeta, abstractmethod
from inspect import signature
import numpy as np
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    _fit_context,
    clone,
    is_classifier,
)
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import (
    _check_multimetric_scoring,
    _MultimetricScorer,
    get_scorer_names,
)
from sklearn.utils import Bunch, check_random_state
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)

class BaseSearchCV(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """Abstract base class for hyper parameter search with cross-validation."""
    _parameter_constraints: dict = {'estimator': [HasMethods(['fit'])], 'scoring': [StrOptions(set(get_scorer_names())), callable, list, tuple, dict, None], 'n_jobs': [numbers.Integral, None], 'refit': ['boolean', str, callable], 'cv': ['cv_object'], 'verbose': ['verbose'], 'pre_dispatch': [numbers.Integral, str], 'error_score': [StrOptions({'raise'}), numbers.Real], 'return_train_score': ['boolean']}

    @abstractmethod
    def __init__(self, estimator, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=True):
        self.scoring = scoring
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

    def _check_refit_for_multimetric(self, scores):
        """Check `refit` is compatible with `scores` is valid"""
        multimetric_refit_msg = f'For multi-metric scoring, the parameter refit must be set to a scorer key or a callable to refit an estimator with the best parameter setting on the whole data and make the best_* attributes available for that metric. If this is not needed, refit should be set to False explicitly. {self.refit!r} was passed.'
        valid_refit_dict = isinstance(self.refit, str) and self.refit in scores
        if self.refit is not False and (not valid_refit_dict) and (not callable(self.refit)):
            raise ValueError(multimetric_refit_msg)

    def _get_scorers(self):
        """Get the scorer(s) to be used.

        This is used in ``fit`` and ``get_metadata_routing``.

        Returns
        -------
        scorers, refit_metric
        """
        refit_metric = 'score'
        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit
            scorers = _MultimetricScorer(scorers=scorers, raise_exc=self.error_score == 'raise')
        return (scorers, refit_metric)

    def _check_scorers_accept_sample_weight(self):
        scorers, _ = self._get_scorers()
        if isinstance(scorers, _MultimetricScorer):
            for name, scorer in scorers._scorers.items():
                if not scorer._accept_sample_weight():
                    warnings.warn(f'The scoring {name}={scorer} does not support sample_weight, which may lead to statistically incorrect results when fitting {self} with sample_weight. ')
            return scorers._accept_sample_weight()
        if hasattr(scorers, '_accept_sample_weight'):
            accept = scorers._accept_sample_weight()
        else:
            accept = 'sample_weight' in signature(scorers).parameters
        if not accept:
            warnings.warn(f'The scoring {scorers} does not support sample_weight, which may lead to statistically incorrect results when fitting {self} with sample_weight. ')
        return accept

    def _get_routed_params_for_fit(self, params):
        """Get the parameters to be used for routing.

        This is a method instead of a snippet in ``fit`` since it's used twice,
        here in ``fit``, and in ``HalvingRandomSearchCV.fit``.
        """
        if _routing_enabled():
            routed_params = process_routing(self, 'fit', **params)
        else:
            params = params.copy()
            groups = params.pop('groups', None)
            routed_params = Bunch(estimator=Bunch(fit=params), splitter=Bunch(split={'groups': groups}), scorer=Bunch(score={}))
            if params.get('sample_weight') is not None and self._check_scorers_accept_sample_weight():
                routed_params.scorer.score['sample_weight'] = params['sample_weight']
        return routed_params
