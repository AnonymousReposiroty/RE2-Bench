import array
import warnings
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import _num_samples, check_array, check_is_fitted

class MultiLabelBinarizer(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None):
    """Transform between iterable of iterables and a multilabel format.

    Although a list of sets or tuples is a very intuitive format for multilabel
    data, it is unwieldy to process. This transformer converts between this
    intuitive format and the supported multilabel format: a (samples x classes)
    binary matrix indicating the presence of a class label.

    Parameters
    ----------
    classes : array-like of shape (n_classes,), default=None
        Indicates an ordering for the class labels.
        All entries should be unique (cannot contain duplicate classes).

    sparse_output : bool, default=False
        Set to True if output binary array is desired in CSR sparse format.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        A copy of the `classes` parameter when provided.
        Otherwise it corresponds to the sorted set of classes found
        when fitting.

    See Also
    --------
    OneHotEncoder : Encode categorical features using a one-hot aka one-of-K
        scheme.

    Examples
    --------
    >>> from sklearn.preprocessing import MultiLabelBinarizer
    >>> mlb = MultiLabelBinarizer()
    >>> mlb.fit_transform([(1, 2), (3,)])
    array([[1, 1, 0],
           [0, 0, 1]])
    >>> mlb.classes_
    array([1, 2, 3])

    >>> mlb.fit_transform([{'sci-fi', 'thriller'}, {'comedy'}])
    array([[0, 1, 1],
           [1, 0, 0]])
    >>> list(mlb.classes_)
    ['comedy', 'sci-fi', 'thriller']

    A common mistake is to pass in a list, which leads to the following issue:

    >>> mlb = MultiLabelBinarizer()
    >>> mlb.fit(['sci-fi', 'thriller', 'comedy'])
    MultiLabelBinarizer()
    >>> mlb.classes_
    array(['-', 'c', 'd', 'e', 'f', 'h', 'i', 'l', 'm', 'o', 'r', 's', 't',
        'y'], dtype=object)

    To correct this, the list of labels should be passed in as:

    >>> mlb = MultiLabelBinarizer()
    >>> mlb.fit([['sci-fi', 'thriller', 'comedy']])
    MultiLabelBinarizer()
    >>> mlb.classes_
    array(['comedy', 'sci-fi', 'thriller'], dtype=object)
    """
    _parameter_constraints: dict = {'classes': ['array-like', None], 'sparse_output': ['boolean']}

    def __init__(self, *, classes=None, sparse_output=False):
        self.classes = classes
        self.sparse_output = sparse_output

    def transform(self, y):
        """Transform the given label sets.

        Parameters
        ----------
        y : iterable of iterables
            A set of labels (any orderable and hashable object) for each
            sample. If the `classes` parameter is set, `y` will not be
            iterated.

        Returns
        -------
        y_indicator : array or CSR matrix, shape (n_samples, n_classes)
            A matrix such that `y_indicator[i, j] = 1` iff `classes_[j]` is in
            `y[i]`, and 0 otherwise.
        """
        check_is_fitted(self)
        class_to_index = self._build_cache()
        yt = self._transform(y, class_to_index)
        if not self.sparse_output:
            yt = yt.toarray()
        return yt

    def _build_cache(self):
        if self._cached_dict is None:
            self._cached_dict = dict(zip(self.classes_, range(len(self.classes_))))
        return self._cached_dict

    def _transform(self, y, class_mapping):
        """Transforms the label sets with a given mapping.

        Parameters
        ----------
        y : iterable of iterables
            A set of labels (any orderable and hashable object) for each
            sample. If the `classes` parameter is set, `y` will not be
            iterated.

        class_mapping : Mapping
            Maps from label to column index in label indicator matrix.

        Returns
        -------
        y_indicator : sparse matrix of shape (n_samples, n_classes)
            Label indicator matrix. Will be of CSR format.
        """
        indices = array.array('i')
        indptr = array.array('i', [0])
        unknown = set()
        for labels in y:
            index = set()
            for label in labels:
                try:
                    index.add(class_mapping[label])
                except KeyError:
                    unknown.add(label)
            indices.extend(index)
            indptr.append(len(indices))
        if unknown:
            warnings.warn('unknown class(es) {0} will be ignored'.format(sorted(unknown, key=str)))
        data = np.ones(len(indices), dtype=int)
        return sp.csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, len(class_mapping)))
