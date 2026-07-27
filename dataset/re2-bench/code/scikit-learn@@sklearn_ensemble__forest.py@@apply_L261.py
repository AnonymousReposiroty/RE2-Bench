from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.sparse import issparse
from sklearn.base import (
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
    is_classifier,
)
from sklearn.ensemble._base import BaseEnsemble, _partition_estimators
from sklearn.tree._tree import DOUBLE, DTYPE
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_feature_names_in,
    _check_sample_weight,
    _num_samples,
    check_is_fitted,
    validate_data,
)

class BaseForest(MultiOutputMixin, BaseEnsemble, metaclass=ABCMeta):
    """
    Base class for forests of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    _parameter_constraints: dict = {'n_estimators': [Interval(Integral, 1, None, closed='left')], 'bootstrap': ['boolean'], 'oob_score': ['boolean', callable], 'n_jobs': [Integral, None], 'random_state': ['random_state'], 'verbose': ['verbose'], 'warm_start': ['boolean'], 'max_samples': [None, Interval(RealNotInt, 0.0, 1.0, closed='right'), Interval(Integral, 1, None, closed='left')]}

    @abstractmethod
    def __init__(self, estimator, n_estimators=100, *, estimator_params=tuple(), bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, max_samples=None):
        super().__init__(estimator=estimator, n_estimators=n_estimators, estimator_params=estimator_params)
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.max_samples = max_samples

    def apply(self, X):
        """
        Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        X = self._validate_X_predict(X)
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='threads')((delayed(tree.apply)(X, check_input=False) for tree in self.estimators_))
        return np.array(results).T

    def _validate_X_predict(self, X):
        """
        Validate X whenever one tries to predict, apply, predict_proba."""
        check_is_fitted(self)
        if self.estimators_[0]._support_missing_values(X):
            ensure_all_finite = 'allow-nan'
        else:
            ensure_all_finite = True
        X = validate_data(self, X, dtype=DTYPE, accept_sparse='csr', reset=False, ensure_all_finite=ensure_all_finite)
        if issparse(X) and (X.indices.dtype != np.intc or X.indptr.dtype != np.intc):
            raise ValueError('No support for np.int64 index based sparse matrices')
        return X
