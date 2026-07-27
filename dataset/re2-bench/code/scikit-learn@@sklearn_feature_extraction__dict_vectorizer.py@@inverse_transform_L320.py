import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils import check_array, metadata_routing
from sklearn.utils.validation import check_is_fitted

class DictVectorizer(TransformerMixin, BaseEstimator):
    """Transforms lists of feature-value mappings to vectors.

    This transformer turns lists of mappings (dict-like objects) of feature
    names to feature values into Numpy arrays or scipy.sparse matrices for use
    with scikit-learn estimators.

    When feature values are strings, this transformer will do a binary one-hot
    (aka one-of-K) coding: one boolean-valued feature is constructed for each
    of the possible string values that the feature can take on. For instance,
    a feature "f" that can take on the values "ham" and "spam" will become two
    features in the output, one signifying "f=ham", the other "f=spam".

    If a feature value is a sequence or set of strings, this transformer
    will iterate over the values and will count the occurrences of each string
    value.

    However, note that this transformer will only do a binary one-hot encoding
    when feature values are of type string. If categorical features are
    represented as numeric values such as int or iterables of strings, the
    DictVectorizer can be followed by
    :class:`~sklearn.preprocessing.OneHotEncoder` to complete
    binary one-hot encoding.

    Features that do not occur in a sample (mapping) will have a zero value
    in the resulting array/matrix.

    For an efficiency comparison of the different feature extractors, see
    :ref:`sphx_glr_auto_examples_text_plot_hashing_vs_dict_vectorizer.py`.

    Read more in the :ref:`User Guide <dict_feature_extraction>`.

    Parameters
    ----------
    dtype : dtype, default=np.float64
        The type of feature values. Passed to Numpy array/scipy.sparse matrix
        constructors as the dtype argument.
    separator : str, default="="
        Separator string used when constructing new features for one-hot
        coding.
    sparse : bool, default=True
        Whether transform should produce scipy.sparse matrices.
    sort : bool, default=True
        Whether ``feature_names_`` and ``vocabulary_`` should be
        sorted when fitting.

    Attributes
    ----------
    vocabulary_ : dict
        A dictionary mapping feature names to feature indices.

    feature_names_ : list
        A list of length n_features containing the feature names (e.g., "f=ham"
        and "f=spam").

    See Also
    --------
    FeatureHasher : Performs vectorization using only a hash function.
    sklearn.preprocessing.OrdinalEncoder : Handles nominal/categorical
        features encoded as columns of arbitrary data types.

    Examples
    --------
    >>> from sklearn.feature_extraction import DictVectorizer
    >>> v = DictVectorizer(sparse=False)
    >>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
    >>> X = v.fit_transform(D)
    >>> X
    array([[2., 0., 1.],
           [0., 1., 3.]])
    >>> v.inverse_transform(X) == [{'bar': 2.0, 'foo': 1.0},
    ...                            {'baz': 1.0, 'foo': 3.0}]
    True
    >>> v.transform({'foo': 4, 'unseen_feature': 3})
    array([[0., 0., 4.]])
    """
    __metadata_request__inverse_transform = {'dict_type': metadata_routing.UNUSED}
    _parameter_constraints: dict = {'dtype': 'no_validation', 'separator': [str], 'sparse': ['boolean'], 'sort': ['boolean']}

    def __init__(self, *, dtype=np.float64, separator='=', sparse=True, sort=True):
        self.dtype = dtype
        self.separator = separator
        self.sparse = sparse
        self.sort = sort

    def inverse_transform(self, X, dict_type=dict):
        """Transform array or sparse matrix X back to feature mappings.

        X must have been produced by this DictVectorizer's transform or
        fit_transform method; it may only have passed through transformers
        that preserve the number of features and their order.

        In the case of one-hot/one-of-K coding, the constructed feature
        names and values are returned rather than the original ones.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Sample matrix.
        dict_type : type, default=dict
            Constructor for feature mappings. Must conform to the
            collections.Mapping API.

        Returns
        -------
        X_original : list of dict_type objects of shape (n_samples,)
            Feature mappings for the samples in X.
        """
        check_is_fitted(self, 'feature_names_')
        X = check_array(X, accept_sparse=['csr', 'csc'])
        n_samples = X.shape[0]
        names = self.feature_names_
        dicts = [dict_type() for _ in range(n_samples)]
        if sp.issparse(X):
            for i, j in zip(*X.nonzero()):
                dicts[i][names[j]] = X[i, j]
        else:
            for i, d in enumerate(dicts):
                for j, v in enumerate(X[i, :]):
                    if v != 0:
                        d[names[j]] = X[i, j]
        return dicts
