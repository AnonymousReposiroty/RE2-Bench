import threading
from numbers import Integral, Real
import numpy as np
from sklearn.base import OutlierMixin, _fit_context
from sklearn.ensemble._bagging import BaseBagging
from sklearn.utils import check_array, check_random_state, gen_batches
from sklearn.utils._chunking import get_chunk_n_rows
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_sample_weight,
    _num_samples,
    check_is_fitted,
    validate_data,
)

class IsolationForest(OutlierMixin, BaseBagging):
    """
    Isolation Forest Algorithm.

    Return the anomaly score of each sample using the IsolationForest algorithm

    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.

    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.

    Read more in the :ref:`User Guide <isolation_forest>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators in the ensemble.

    max_samples : "auto", int or float, default="auto"
        The number of samples to draw from X to train each base estimator.

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.
        - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.

        - If 'auto', the threshold is determined as in the
          original paper.
        - If float, the contamination should be in the range (0, 0.5].

        .. versionchanged:: 0.22
           The default value of ``contamination`` changed from 0.1
           to ``'auto'``.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.

        - If int, then draw `max_features` features.
        - If float, then draw `max(1, int(max_features * n_features_in_))` features.

        Note: using a float number less than 1.0 or integer less than number of
        features will enable feature subsampling and leads to a longer runtime.

    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    n_jobs : int, default=None
        The number of jobs to run in parallel for :meth:`fit`. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using
        all processors. See :term:`Glossary <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : int, default=0
        Controls the verbosity of the tree building process.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.21

    Attributes
    ----------
    estimator_ : :class:`~sklearn.tree.ExtraTreeRegressor` instance
        The child estimator template used to create the collection of
        fitted sub-estimators.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    estimators_ : list of ExtraTreeRegressor instances
        The collection of fitted sub-estimators.

    estimators_features_ : list of ndarray
        The subset of drawn features for each base estimator.

    estimators_samples_ : list of ndarray
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    max_samples_ : int
        The actual number of samples.

    offset_ : float
        Offset used to define the decision function from the raw scores. We
        have the relation: ``decision_function = score_samples - offset_``.
        ``offset_`` is defined as follows. When the contamination parameter is
        set to "auto", the offset is equal to -0.5 as the scores of inliers are
        close to 0 and the scores of outliers are close to -1. When a
        contamination parameter different than "auto" is provided, the offset
        is defined in such a way we obtain the expected number of outliers
        (samples with decision function < 0) in training.

        .. versionadded:: 0.20

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.covariance.EllipticEnvelope : An object for detecting outliers in a
        Gaussian distributed dataset.
    sklearn.svm.OneClassSVM : Unsupervised Outlier Detection.
        Estimate the support of a high-dimensional distribution.
        The implementation is based on libsvm.
    sklearn.neighbors.LocalOutlierFactor : Unsupervised Outlier Detection
        using Local Outlier Factor (LOF).

    Notes
    -----
    The implementation is based on an ensemble of ExtraTreeRegressor. The
    maximum depth of each tree is set to ``ceil(log_2(n))`` where
    :math:`n` is the number of samples used to build the tree
    (see [1]_ for more details).

    References
    ----------
    .. [1] F. T. Liu, K. M. Ting and Z. -H. Zhou.
           :doi:`"Isolation forest." <10.1109/ICDM.2008.17>`
           2008 Eighth IEEE International Conference on Data Mining (ICDM),
           2008, pp. 413-422.
    .. [2] F. T. Liu, K. M. Ting and Z. -H. Zhou.
           :doi:`"Isolation-based anomaly detection."
           <10.1145/2133360.2133363>` ACM Transactions on
           Knowledge Discovery from Data (TKDD) 6.1 (2012): 1-39.

    Examples
    --------
    >>> from sklearn.ensemble import IsolationForest
    >>> X = [[-1.1], [0.3], [0.5], [100]]
    >>> clf = IsolationForest(random_state=0).fit(X)
    >>> clf.predict([[0.1], [0], [90]])
    array([ 1,  1, -1])

    For an example of using isolation forest for anomaly detection see
    :ref:`sphx_glr_auto_examples_ensemble_plot_isolation_forest.py`.
    """
    _parameter_constraints: dict = {'n_estimators': [Interval(Integral, 1, None, closed='left')], 'max_samples': [StrOptions({'auto'}), Interval(Integral, 1, None, closed='left'), Interval(RealNotInt, 0, 1, closed='right')], 'contamination': [StrOptions({'auto'}), Interval(Real, 0, 0.5, closed='right')], 'max_features': [Integral, Interval(Real, 0, 1, closed='right')], 'bootstrap': ['boolean'], 'n_jobs': [Integral, None], 'random_state': ['random_state'], 'verbose': ['verbose'], 'warm_start': ['boolean']}

    def __init__(self, *, n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0, bootstrap=False, n_jobs=None, random_state=None, verbose=0, warm_start=False):
        super().__init__(estimator=None, bootstrap=bootstrap, bootstrap_features=False, n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, warm_start=warm_start, n_jobs=n_jobs, random_state=random_state, verbose=verbose)
        self.contamination = contamination

    def _compute_chunked_score_samples(self, X):
        n_samples = _num_samples(X)
        if self._max_features == X.shape[1]:
            subsample_features = False
        else:
            subsample_features = True
        chunk_n_rows = get_chunk_n_rows(row_bytes=16 * self._max_features, max_n_rows=n_samples)
        slices = gen_batches(n_samples, chunk_n_rows)
        scores = np.zeros(n_samples, order='f')
        for sl in slices:
            scores[sl] = self._compute_score_samples(X[sl], subsample_features)
        return scores

    def _compute_score_samples(self, X, subsample_features):
        """
        Compute the score of each samples in X going through the extra trees.

        Parameters
        ----------
        X : array-like or sparse matrix
            Data matrix.

        subsample_features : bool
            Whether features should be subsampled.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The score of each sample in X.
        """
        n_samples = X.shape[0]
        depths = np.zeros(n_samples, order='f')
        average_path_length_max_samples = _average_path_length([self._max_samples])
        lock = threading.Lock()
        Parallel(verbose=self.verbose, require='sharedmem')((delayed(_parallel_compute_tree_depths)(tree, X, features if subsample_features else None, self._decision_path_lengths[tree_idx], self._average_path_length_per_tree[tree_idx], depths, lock) for tree_idx, (tree, features) in enumerate(zip(self.estimators_, self.estimators_features_))))
        denominator = len(self.estimators_) * average_path_length_max_samples
        scores = 2 ** (-np.divide(depths, denominator, out=np.ones_like(depths), where=denominator != 0))
        return scores
