from collections import defaultdict
from collections.abc import Mapping
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils import metadata_routing
from sklearn.utils._param_validation import HasMethods, Interval, RealNotInt, StrOptions
from sklearn.utils.fixes import _IS_32BIT

class CountVectorizer(_VectorizerMixin, BaseEstimator):
    """Convert a collection of text documents to a matrix of token counts.

    This implementation produces a sparse representation of the counts using
    scipy.sparse.csr_matrix.

    If you do not provide an a-priori dictionary and you do not use an analyzer
    that does some kind of feature selection then the number of features will
    be equal to the vocabulary size found by analyzing the data.

    For an efficiency comparison of the different feature extractors, see
    :ref:`sphx_glr_auto_examples_text_plot_hashing_vs_dict_vectorizer.py`.

    Read more in the :ref:`User Guide <text_feature_extraction>`.

    Parameters
    ----------
    input : {'filename', 'file', 'content'}, default='content'
        - If `'filename'`, the sequence passed as an argument to fit is
          expected to be a list of filenames that need reading to fetch
          the raw content to analyze.

        - If `'file'`, the sequence items must have a 'read' method (file-like
          object) that is called to fetch the bytes in memory.

        - If `'content'`, the input is expected to be a sequence of items that
          can be of type string or byte.

    encoding : str, default='utf-8'
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}, default='strict'
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode'} or callable, default=None
        Remove accents and perform other character normalization
        during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        a direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) means no character normalization is performed.

        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.

    lowercase : bool, default=True
        Convert all characters to lowercase before tokenizing.

    preprocessor : callable, default=None
        Override the preprocessing (strip_accents and lowercase) stage while
        preserving the tokenizing and n-grams generation steps.
        Only applies if ``analyzer`` is not callable.

    tokenizer : callable, default=None
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.

    stop_words : {'english'}, list, default=None
        If 'english', a built-in stop word list for English is used.
        There are several known issues with 'english' and you should
        consider an alternative (see :ref:`stop_words`).

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.

        If None, no stop words will be used. In this case, setting `max_df`
        to a higher value, such as in the range (0.7, 1.0), can automatically detect
        and filter stop words based on intra corpus document frequency of terms.

    token_pattern : str or None, default=r"(?u)\\\\b\\\\w\\\\w+\\\\b"
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp select tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

        If there is a capturing group in token_pattern then the
        captured group content, not the entire match, becomes the token.
        At most one capturing group is permitted.

    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        word n-grams or char n-grams to be extracted. All values of n such
        such that min_n <= n <= max_n will be used. For example an
        ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
        unigrams and bigrams, and ``(2, 2)`` means only bigrams.
        Only applies if ``analyzer`` is not callable.

    analyzer : {'word', 'char', 'char_wb'} or callable, default='word'
        Whether the feature should be made of word n-gram or character
        n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

        .. versionchanged:: 0.21

        Since v0.21, if ``input`` is ``filename`` or ``file``, the data is
        first read from the file and then passed to the given callable
        analyzer.

    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : int, default=None
        If not None, build a vocabulary that only consider the top
        `max_features` ordered by term frequency across the corpus.
        Otherwise, all features are used.

        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable, default=None
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.

    binary : bool, default=False
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.

    dtype : dtype, default=np.int64
        Type of the matrix returned by fit_transform() or transform().

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.

    fixed_vocabulary_ : bool
        True if a fixed vocabulary of term to indices mapping
        is provided by the user.

    See Also
    --------
    HashingVectorizer : Convert a collection of text documents to a
        matrix of token counts.

    TfidfVectorizer : Convert a collection of raw documents to a matrix
        of TF-IDF features.

    Examples
    --------
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> corpus = [
    ...     'This is the first document.',
    ...     'This document is the second document.',
    ...     'And this is the third one.',
    ...     'Is this the first document?',
    ... ]
    >>> vectorizer = CountVectorizer()
    >>> X = vectorizer.fit_transform(corpus)
    >>> vectorizer.get_feature_names_out()
    array(['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third',
           'this'], ...)
    >>> print(X.toarray())
    [[0 1 1 1 0 0 1 0 1]
     [0 2 0 1 0 1 1 0 1]
     [1 0 0 1 1 0 1 1 1]
     [0 1 1 1 0 0 1 0 1]]
    >>> vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
    >>> X2 = vectorizer2.fit_transform(corpus)
    >>> vectorizer2.get_feature_names_out()
    array(['and this', 'document is', 'first document', 'is the', 'is this',
           'second document', 'the first', 'the second', 'the third', 'third one',
           'this document', 'this is', 'this the'], ...)
     >>> print(X2.toarray())
     [[0 0 1 1 0 0 1 0 0 0 0 1 0]
     [0 1 0 1 0 1 0 1 0 0 1 0 0]
     [1 0 0 1 0 0 0 0 1 1 0 1 0]
     [0 0 1 0 1 0 1 0 0 0 0 0 1]]
    """
    __metadata_request__fit = {'raw_documents': metadata_routing.UNUSED}
    __metadata_request__transform = {'raw_documents': metadata_routing.UNUSED}
    _parameter_constraints: dict = {'input': [StrOptions({'filename', 'file', 'content'})], 'encoding': [str], 'decode_error': [StrOptions({'strict', 'ignore', 'replace'})], 'strip_accents': [StrOptions({'ascii', 'unicode'}), None, callable], 'lowercase': ['boolean'], 'preprocessor': [callable, None], 'tokenizer': [callable, None], 'stop_words': [StrOptions({'english'}), list, None], 'token_pattern': [str, None], 'ngram_range': [tuple], 'analyzer': [StrOptions({'word', 'char', 'char_wb'}), callable], 'max_df': [Interval(RealNotInt, 0, 1, closed='both'), Interval(Integral, 1, None, closed='left')], 'min_df': [Interval(RealNotInt, 0, 1, closed='both'), Interval(Integral, 1, None, closed='left')], 'max_features': [Interval(Integral, 1, None, closed='left'), None], 'vocabulary': [Mapping, HasMethods('__iter__'), None], 'binary': ['boolean'], 'dtype': 'no_validation'}

    def __init__(self, *, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=np.int64):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary = vocabulary
        self.binary = binary
        self.dtype = dtype

    def _count_vocab(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False"""
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__
        analyze = self.build_analyzer()
        j_indices = []
        indptr = []
        values = _make_int_array()
        indptr.append(0)
        for doc in raw_documents:
            feature_counter = {}
            for feature in analyze(doc):
                try:
                    feature_idx = vocabulary[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                except KeyError:
                    continue
            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))
        if not fixed_vocab:
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError('empty vocabulary; perhaps the documents only contain stop words')
        if indptr[-1] > np.iinfo(np.int32).max:
            if _IS_32BIT:
                raise ValueError('sparse CSR array has {} non-zero elements and requires 64 bit indexing, which is unsupported with 32 bit Python.'.format(indptr[-1]))
            indices_dtype = np.int64
        else:
            indices_dtype = np.int32
        j_indices = np.asarray(j_indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        values = np.frombuffer(values, dtype=np.intc)
        X = sp.csr_matrix((values, j_indices, indptr), shape=(len(indptr) - 1, len(vocabulary)), dtype=self.dtype)
        X.sort_indices()
        return (vocabulary, X)

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.

        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        """
        if isinstance(raw_documents, str):
            raise ValueError('Iterable over raw text documents expected, string object received.')
        self._check_vocabulary()
        _, X = self._count_vocab(raw_documents, fixed_vocab=True)
        if self.binary:
            X.data.fill(1)
        return X
