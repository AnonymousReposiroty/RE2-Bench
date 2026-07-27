import warnings
import numpy as np
from sklearn.utils import (
    _safe_indexing,
    check_random_state,
    indexable,
    metadata_routing,
)
from sklearn.utils._array_api import (
    _convert_to_numpy,
    get_namespace,
    get_namespace_and_device,
    move_to,
)
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples, check_array, column_or_1d

class StratifiedKFold(_BaseKFold):
    """Class-wise stratified K-Fold cross-validator.

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a variation of KFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples for each class in `y` in a binary or multiclass classification
    setting.

    Read more in the :ref:`User Guide <stratified_k_fold>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    .. note::

        Stratification on the class label solves an engineering problem rather
        than a statistical one. See :ref:`stratification` for more details.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.

    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> skf = StratifiedKFold(n_splits=2)
    >>> skf.get_n_splits()
    2
    >>> print(skf)
    StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    >>> for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[1 3]
      Test:  index=[0 2]
    Fold 1:
      Train: index=[0 2]
      Test:  index=[1 3]

    Notes
    -----
    The implementation is designed to:

    * Generate test sets such that all contain the same distribution of
      classes, or as close as possible.
    * Be invariant to class label: relabelling ``y = ["Happy", "Sad"]`` to
      ``y = [1, 0]`` should not change the indices generated.
    * Preserve order dependencies in the dataset ordering, when
      ``shuffle=False``: all samples from class k in some test set were
      contiguous in y, or separated in y by samples from classes other than k.
    * Generate test sets where the smallest and largest differ by at most one
      sample.

    .. versionchanged:: 0.22
        The previous implementation did not follow the last constraint.

    See Also
    --------
    RepeatedStratifiedKFold : Repeats Stratified K-Fold n times.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _make_test_folds(self, X, y=None):
        rng = check_random_state(self.random_state)
        xp, is_array_api = get_namespace(y)
        if is_array_api:
            y = _convert_to_numpy(y, xp)
        else:
            y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError('Supported target types are: {}. Got {!r} instead.'.format(allowed_target_types, type_of_target_y))
        y = column_or_1d(y)
        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]
        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError('n_splits=%d cannot be greater than the number of members in each class.' % self.n_splits)
        if self.n_splits > min_groups:
            warnings.warn('The least populated class in y has only %d members, which is less than n_splits=%d.' % (min_groups, self.n_splits), UserWarning)
        y_order = np.sort(y_encoded)
        allocation = np.asarray([np.bincount(y_order[i::self.n_splits], minlength=n_classes) for i in range(self.n_splits)])
        test_folds = np.empty(len(y), dtype='i')
        for k in range(n_classes):
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                rng.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class
        return test_folds
