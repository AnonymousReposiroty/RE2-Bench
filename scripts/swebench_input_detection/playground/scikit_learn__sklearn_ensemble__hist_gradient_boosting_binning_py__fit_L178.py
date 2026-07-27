import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble._hist_gradient_boosting.common import (
    ALMOST_INF,
    X_BINNED_DTYPE,
    X_BITSET_INNER_DTYPE,
    X_DTYPE,
)
from sklearn.utils import check_array, check_random_state
from sklearn.utils.parallel import Parallel, delayed

class _BinMapper(TransformerMixin, BaseEstimator):
    """Transformer that maps a dataset into integer-valued bins.

    For continuous features, the bins are created in a feature-wise fashion,
    using quantiles so that each bins contains approximately the same number
    of samples. For large datasets, quantiles are computed on a subset of the
    data to speed-up the binning, but the quantiles should remain stable.

    For categorical features, the raw categorical values are expected to be
    in [0, 254] (this is not validated here though) and each category
    corresponds to a bin. All categorical values must be known at
    initialization: transform() doesn't know how to bin unknown categorical
    values. Note that transform() is only used on non-training data in the
    case of early stopping.

    Features with a small number of values may be binned into less than
    ``n_bins`` bins. The last bin (at index ``n_bins - 1``) is always reserved
    for missing values.

    Parameters
    ----------
    n_bins : int, default=256
        The maximum number of bins to use (including the bin for missing
        values). Should be in [3, 256]. Non-missing values are binned on
        ``max_bins = n_bins - 1`` bins. The last bin is always reserved for
        missing values. If for a given feature the number of unique values is
        less than ``max_bins``, then those unique values will be used to
        compute the bin thresholds, instead of the quantiles. For categorical
        features indicated by ``is_categorical``, the docstring for
        ``is_categorical`` details on this procedure.
    subsample : int or None, default=2e5
        If ``n_samples > subsample``, then ``sub_samples`` samples will be
        randomly chosen to compute the quantiles. If ``None``, the whole data
        is used.
    is_categorical : ndarray of bool of shape (n_features,), default=None
        Indicates categorical features. By default, all features are
        considered continuous.
    known_categories : list of {ndarray, None} of shape (n_features,),             default=none
        For each categorical feature, the array indicates the set of unique
        categorical values. These should be the possible values over all the
        data, not just the training data. For continuous features, the
        corresponding entry should be None.
    random_state: int, RandomState instance or None, default=None
        Pseudo-random number generator to control the random sub-sampling.
        Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.
    n_threads : int, default=None
        Number of OpenMP threads to use. `_openmp_effective_n_threads` is called
        to determine the effective number of threads use, which takes cgroups CPU
        quotes into account. See the docstring of `_openmp_effective_n_threads`
        for details.

    Attributes
    ----------
    bin_thresholds_ : list of ndarray
        For each feature, each array indicates how to map a feature into a
        binned feature. The semantic and size depends on the nature of the
        feature:
        - for real-valued features, the array corresponds to the real-valued
          bin thresholds (the upper bound of each bin). There are ``max_bins
          - 1`` thresholds, where ``max_bins = n_bins - 1`` is the number of
          bins used for non-missing values.
        - for categorical features, the array is a map from a binned category
          value to the raw category value. The size of the array is equal to
          ``min(max_bins, category_cardinality)`` where we ignore missing
          values in the cardinality.
    n_bins_non_missing_ : ndarray, dtype=np.uint32
        For each feature, gives the number of bins actually used for
        non-missing values. For features with a lot of unique values, this is
        equal to ``n_bins - 1``.
    is_categorical_ : ndarray of shape (n_features,), dtype=np.uint8
        Indicator for categorical features.
    missing_values_bin_idx_ : np.uint8
        The index of the bin where missing values are mapped. This is a
        constant across all features. This corresponds to the last bin, and
        it is always equal to ``n_bins - 1``. Note that if ``n_bins_non_missing_``
        is less than ``n_bins - 1`` for a given feature, then there are
        empty (and unused) bins.
    """

    def __init__(self, n_bins=256, subsample=int(200000.0), is_categorical=None, known_categories=None, random_state=None, n_threads=None):
        self.n_bins = n_bins
        self.subsample = subsample
        self.is_categorical = is_categorical
        self.known_categories = known_categories
        self.random_state = random_state
        self.n_threads = n_threads

    def fit(self, X, y=None):
        """Fit data X by computing the binning thresholds.

        The last bin is reserved for missing values, whether missing values
        are present in the data or not.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to bin.
        y: None
            Ignored.

        Returns
        -------
        self : object
        """
        if not 3 <= self.n_bins <= 256:
            raise ValueError('n_bins={} should be no smaller than 3 and no larger than 256.'.format(self.n_bins))
        X = check_array(X, dtype=[X_DTYPE], ensure_all_finite=False)
        max_bins = self.n_bins - 1
        rng = check_random_state(self.random_state)
        if self.subsample is not None and X.shape[0] > self.subsample:
            subset = rng.choice(X.shape[0], self.subsample, replace=False)
            X = X.take(subset, axis=0)
        if self.is_categorical is None:
            self.is_categorical_ = np.zeros(X.shape[1], dtype=np.uint8)
        else:
            self.is_categorical_ = np.asarray(self.is_categorical, dtype=np.uint8)
        n_features = X.shape[1]
        known_categories = self.known_categories
        if known_categories is None:
            known_categories = [None] * n_features
        for f_idx in range(n_features):
            is_categorical = self.is_categorical_[f_idx]
            known_cats = known_categories[f_idx]
            if is_categorical and known_cats is None:
                raise ValueError(f'Known categories for feature {f_idx} must be provided.')
            if not is_categorical and known_cats is not None:
                raise ValueError(f"Feature {f_idx} isn't marked as a categorical feature, but categories were passed.")
        self.missing_values_bin_idx_ = self.n_bins - 1
        self.bin_thresholds_ = [None] * n_features
        n_bins_non_missing = [None] * n_features
        non_cat_thresholds = Parallel(n_jobs=self.n_threads, backend='threading')((delayed(_find_binning_thresholds)(X[:, f_idx], max_bins) for f_idx in range(n_features) if not self.is_categorical_[f_idx]))
        non_cat_idx = 0
        for f_idx in range(n_features):
            if self.is_categorical_[f_idx]:
                thresholds = known_categories[f_idx]
                n_bins_non_missing[f_idx] = thresholds.shape[0]
                self.bin_thresholds_[f_idx] = thresholds
            else:
                self.bin_thresholds_[f_idx] = non_cat_thresholds[non_cat_idx]
                n_bins_non_missing[f_idx] = self.bin_thresholds_[f_idx].shape[0] + 1
                non_cat_idx += 1
        self.n_bins_non_missing_ = np.array(n_bins_non_missing, dtype=np.uint32)
        return self

def test_input(pred_input):
    obj_ins = _BinMapper(n_bins = 11, subsample = 200000, is_categorical = None, known_categories = None, random_state = 42, n_threads = None)
    obj_ins_pred = _BinMapper(n_bins = pred_input['self']['n_bins'], subsample = pred_input['self']['subsample'], is_categorical = pred_input['self']['is_categorical'], known_categories = pred_input['self']['known_categories'], random_state = pred_input['self']['random_state'], n_threads = pred_input['self']['n_threads'])
    assert obj_ins.fit(X = np.array([[ 0.49671415, 9.99861736], [ 0.64768854, 10.0152303 ], [-0.23415337, 9.99765863], ..., [-0.71920925, 10.00807964], [-0.10042671, 10.00237009], [-0.12582283, 10.01781749]], shape=(1000000, 2)), y = None)==obj_ins_pred.fit(X = pred_input['args']['X'], y = pred_input['args']['y']), 'Prediction failed!'