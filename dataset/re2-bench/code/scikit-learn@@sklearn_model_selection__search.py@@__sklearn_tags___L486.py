import numbers
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import numpy as np
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
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils._tags import get_tags

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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        sub_estimator_tags = get_tags(self.estimator)
        tags.estimator_type = sub_estimator_tags.estimator_type
        tags.classifier_tags = deepcopy(sub_estimator_tags.classifier_tags)
        tags.regressor_tags = deepcopy(sub_estimator_tags.regressor_tags)
        tags.input_tags.pairwise = sub_estimator_tags.input_tags.pairwise
        tags.input_tags.sparse = sub_estimator_tags.input_tags.sparse
        tags.array_api_support = sub_estimator_tags.array_api_support
        return tags
