from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils import check_array, check_random_state, gen_batches, gen_even_slices
from sklearn.utils.validation import check_is_fitted, validate_data

class _BaseSparseCoding(ClassNamePrefixFeaturesOutMixin, TransformerMixin):
    """Base class from SparseCoder and DictionaryLearning algorithms."""

    def __init__(self, transform_algorithm, transform_n_nonzero_coefs, transform_alpha, split_sign, n_jobs, positive_code, transform_max_iter):
        self.transform_algorithm = transform_algorithm
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.transform_alpha = transform_alpha
        self.transform_max_iter = transform_max_iter
        self.split_sign = split_sign
        self.n_jobs = n_jobs
        self.positive_code = positive_code

    def _inverse_transform(self, code, dictionary):
        """Private method allowing to accommodate both DictionaryLearning and
        SparseCoder."""
        code = check_array(code)
        expected_n_components = dictionary.shape[0]
        if self.split_sign:
            expected_n_components += expected_n_components
        if not code.shape[1] == expected_n_components:
            raise ValueError(f'The number of components in the code is different from the number of components in the dictionary.Expected {expected_n_components}, got {code.shape[1]}.')
        if self.split_sign:
            n_samples, n_features = code.shape
            n_features //= 2
            code = code[:, :n_features] - code[:, n_features:]
        return code @ dictionary

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Data to be transformed back. Must have the same number of
            components as the data used to train the model.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)
        return self._inverse_transform(X, self.components_)
