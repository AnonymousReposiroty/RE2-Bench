import numpy as np
import scipy.sparse as sp
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.preprocessing import normalize
from sklearn.utils._param_validation import HasMethods, Interval, RealNotInt, StrOptions
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    check_array,
    check_is_fitted,
    validate_data,
)

class TfidfTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator, auto_wrap_output_keys=None):
    """Transform a count matrix to a normalized tf or tf-idf representation.

    Tf means term-frequency while tf-idf means term-frequency times inverse
    document-frequency. This is a common term weighting scheme in information
    retrieval, that has also found good use in document classification.

    The goal of using tf-idf instead of the raw frequencies of occurrence of a
    token in a given document is to scale down the impact of tokens that occur
    very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training
    corpus.

    The formula that is used to compute the tf-idf for a term t of a document d
    in a document set is tf-idf(t, d) = tf(t, d) * idf(t), and the idf is
    computed as idf(t) = log [ n / df(t) ] + 1 (if ``smooth_idf=False``), where
    n is the total number of documents in the document set and df(t) is the
    document frequency of t; the document frequency is the number of documents
    in the document set that contain the term t. The effect of adding "1" to
    the idf in the equation above is that terms with zero idf, i.e., terms
    that occur in all documents in a training set, will not be entirely
    ignored.
    (Note that the idf formula above differs from the standard textbook
    notation that defines the idf as
    idf(t) = log [ n / (df(t) + 1) ]).

    If ``smooth_idf=True`` (the default), the constant "1" is added to the
    numerator and denominator of the idf as if an extra document was seen
    containing every term in the collection exactly once, which prevents
    zero divisions: idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1.

    Furthermore, the formulas used to compute tf and idf depend
    on parameter settings that correspond to the SMART notation used in IR
    as follows:

    Tf is "n" (natural) by default, "l" (logarithmic) when
    ``sublinear_tf=True``.
    Idf is "t" when use_idf is given, "n" (none) otherwise.
    Normalization is "c" (cosine) when ``norm='l2'``, "n" (none)
    when ``norm=None``.

    Read more in the :ref:`User Guide <text_feature_extraction>`.

    Parameters
    ----------
    norm : {'l1', 'l2'} or None, default='l2'
        Each output row will have unit norm, either:

        - 'l2': Sum of squares of vector elements is 1. The cosine
          similarity between two vectors is their dot product when l2 norm has
          been applied.
        - 'l1': Sum of absolute values of vector elements is 1.
          See :func:`~sklearn.preprocessing.normalize`.
        - None: No normalization.

    use_idf : bool, default=True
        Enable inverse-document-frequency reweighting. If False, idf(t) = 1.

    smooth_idf : bool, default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : bool, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Attributes
    ----------
    idf_ : array of shape (n_features)
        The inverse document frequency (IDF) vector; only defined
        if  ``use_idf`` is True.

        .. versionadded:: 0.20

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 1.0

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    CountVectorizer : Transforms text into a sparse matrix of n-gram counts.

    TfidfVectorizer : Convert a collection of raw documents to a matrix of
        TF-IDF features.

    HashingVectorizer : Convert a collection of text documents to a matrix
        of token occurrences.

    References
    ----------
    .. [Yates2011] R. Baeza-Yates and B. Ribeiro-Neto (2011). Modern
                   Information Retrieval. Addison Wesley, pp. 68-74.

    .. [MRS2008] C.D. Manning, P. Raghavan and H. Schütze  (2008).
                   Introduction to Information Retrieval. Cambridge University
                   Press, pp. 118-120.

    Examples
    --------
    >>> from sklearn.feature_extraction.text import TfidfTransformer
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from sklearn.pipeline import Pipeline
    >>> corpus = ['this is the first document',
    ...           'this document is the second document',
    ...           'and this is the third one',
    ...           'is this the first document']
    >>> vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
    ...               'and', 'one']
    >>> pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
    ...                  ('tfid', TfidfTransformer())]).fit(corpus)
    >>> pipe['count'].transform(corpus).toarray()
    array([[1, 1, 1, 1, 0, 1, 0, 0],
           [1, 2, 0, 1, 1, 1, 0, 0],
           [1, 0, 0, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 0, 1, 0, 0]])
    >>> pipe['tfid'].idf_
    array([1.        , 1.22314355, 1.51082562, 1.        , 1.91629073,
           1.        , 1.91629073, 1.91629073])
    >>> pipe.transform(corpus).shape
    (4, 8)
    """
    _parameter_constraints: dict = {'norm': [StrOptions({'l1', 'l2'}), None], 'use_idf': ['boolean'], 'smooth_idf': ['boolean'], 'sublinear_tf': ['boolean']}

    def __init__(self, *, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation.

        Parameters
        ----------
        X : sparse matrix of (n_samples, n_features)
            A matrix of term/token counts.

        copy : bool, default=True
            Whether to copy X and operate on the copy or perform in-place
            operations. `copy=False` will only be effective with CSR sparse matrix.

        Returns
        -------
        vectors : sparse matrix of shape (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse='csr', dtype=[np.float64, np.float32], copy=copy, reset=False)
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=X.dtype)
        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1.0
        if hasattr(self, 'idf_'):
            X.data *= self.idf_[X.indices]
        if self.norm is not None:
            X = normalize(X, norm=self.norm, copy=False)
        return X
