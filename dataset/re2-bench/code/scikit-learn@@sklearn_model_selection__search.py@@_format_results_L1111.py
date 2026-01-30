import numbers
import warnings
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats import rankdata
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    _fit_context,
    clone,
    is_classifier,
)
from sklearn.metrics._scorer import (
    _check_multimetric_scoring,
    _MultimetricScorer,
    get_scorer_names,
)
from sklearn.model_selection._validation import (
    _aggregate_score_dicts,
    _fit_and_score,
    _insert_error_scores,
    _normalize_score_results,
    _warn_or_raise_about_fit_failures,
)
from sklearn.utils._array_api import xpx
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions

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

    def _format_results(self, candidate_params, n_splits, out, more_results=None):
        n_candidates = len(candidate_params)
        out = _aggregate_score_dicts(out)
        results = dict(more_results or {})
        for key, val in results.items():
            results[key] = np.asarray(val)

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)
            if splits:
                for split_idx in range(n_splits):
                    results['split%d_%s' % (split_idx, key_name)] = array[:, split_idx]
            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            if key_name.startswith(('train_', 'test_')) and np.any(~np.isfinite(array_means)):
                warnings.warn(f"One or more of the {key_name.split('_')[0]} scores are non-finite: {array_means}", category=UserWarning)
            array_stds = np.sqrt(np.average((array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds
            if rank:
                if np.isnan(array_means).all():
                    rank_result = np.ones_like(array_means, dtype=np.int32)
                else:
                    min_array_means = np.nanmin(array_means) - 1
                    array_means = xpx.nan_to_num(array_means, fill_value=min_array_means)
                    rank_result = rankdata(-array_means, method='min').astype(np.int32, copy=False)
                results['rank_%s' % key_name] = rank_result
        _store('fit_time', out['fit_time'])
        _store('score_time', out['score_time'])
        for param, ma in _yield_masked_array_for_each_param(candidate_params):
            results[param] = ma
        results['params'] = candidate_params
        test_scores_dict = _normalize_score_results(out['test_scores'])
        if self.return_train_score:
            train_scores_dict = _normalize_score_results(out['train_scores'])
        for scorer_name in test_scores_dict:
            _store('test_%s' % scorer_name, test_scores_dict[scorer_name], splits=True, rank=True, weights=None)
            if self.return_train_score:
                _store('train_%s' % scorer_name, train_scores_dict[scorer_name], splits=True)
        return results
