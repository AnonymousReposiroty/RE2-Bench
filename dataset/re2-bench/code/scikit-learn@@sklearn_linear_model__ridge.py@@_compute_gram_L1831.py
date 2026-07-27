import numpy as np
from scipy import linalg, optimize, sparse
from sklearn.linear_model._base import (
    LinearClassifierMixin,
    LinearModel,
    _preprocess_data,
    _rescale_data,
)
from sklearn.utils._array_api import (
    _is_numpy_namespace,
    _max_precision_float_dtype,
    _ravel,
    device,
    get_namespace,
    get_namespace_and_device,
    move_to,
)
from sklearn.utils.extmath import row_norms, safe_sparse_dot
from sklearn.utils.sparsefuncs import mean_variance_axis

class _RidgeGCV(LinearModel):
    """Ridge regression with built-in Leave-one-out Cross-Validation.

    This class is not intended to be used directly. Use RidgeCV instead.

    `_RidgeGCV` uses a Generalized Cross-Validation for model selection. It's an
    efficient approximation of leave-one-out cross-validation (LOO-CV), where instead of
    computing multiple models by excluding one data point at a time, it uses an
    algebraic shortcut to approximate the LOO-CV error, making it faster and
    computationally more efficient.

    Using a naive grid-search approach with a leave-one-out cross-validation in contrast
    requires to fit `n_samples` models to compute the prediction error for each sample
    and then to repeat this process for each alpha in the grid.

    Here, the prediction error for each sample is computed by solving a **single**
    linear system (in other words a single model) via a matrix factorization (i.e.
    eigendecomposition or SVD) solving the problem stated in the Notes section. Finally,
    we need to repeat this process for each alpha in the grid. The detailed complexity
    is further discussed in Sect. 4 in [1].

    This algebraic approach is only applicable for regularized least squares
    problems. It could potentially be extended to kernel ridge regression.

    See the Notes section and references for more details regarding the formulation
    and the linear system that is solved.

    Notes
    -----

    We want to solve (K + alpha*Id)c = y,
    where K = X X^T is the kernel matrix.

    Let G = (K + alpha*Id).

    Dual solution: c = G^-1y
    Primal solution: w = X^T c

    Compute eigendecomposition K = Q V Q^T.
    Then G^-1 = Q (V + alpha*Id)^-1 Q^T,
    where (V + alpha*Id) is diagonal.
    It is thus inexpensive to inverse for many alphas.

    Let loov be the vector of prediction values for each example
    when the model was fitted with all examples but this example.

    loov = (KG^-1Y - diag(KG^-1)Y) / diag(I-KG^-1)

    Let looe be the vector of prediction errors for each example
    when the model was fitted with all examples but this example.

    looe = y - loov = c / diag(G^-1)

    The best score (negative mean squared error or user-provided scoring) is
    stored in the `best_score_` attribute, and the selected hyperparameter in
    `alpha_`.

    References
    ----------
    [1] http://cbcl.mit.edu/publications/ps/MIT-CSAIL-TR-2007-025.pdf
    [2] https://www.mit.edu/~9.520/spring07/Classes/rlsslides.pdf
    """

    def __init__(self, alphas=(0.1, 1.0, 10.0), *, fit_intercept=True, scoring=None, copy_X=True, gcv_mode=None, store_cv_results=False, is_clf=False, alpha_per_target=False):
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.scoring = scoring
        self.copy_X = copy_X
        self.gcv_mode = gcv_mode
        self.store_cv_results = store_cv_results
        self.is_clf = is_clf
        self.alpha_per_target = alpha_per_target

    def _compute_gram(self, X, sqrt_sw):
        """Computes the Gram matrix XX^T with possible centering.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The preprocessed design matrix.

        sqrt_sw : ndarray of shape (n_samples,)
            square roots of sample weights

        Returns
        -------
        gram : ndarray of shape (n_samples, n_samples)
            The Gram matrix.
        X_mean : ndarray of shape (n_feature,)
            The weighted mean of ``X`` for each feature.

        Notes
        -----
        When X is dense the centering has been done in preprocessing
        so the mean is 0 and we just compute XX^T.

        When X is sparse it has not been centered in preprocessing, but it has
        been scaled by sqrt(sample weights).

        When self.fit_intercept is False no centering is done.

        The centered X is never actually computed because centering would break
        the sparsity of X.
        """
        xp, _ = get_namespace(X)
        center = self.fit_intercept and sparse.issparse(X)
        if not center:
            X_mean = xp.zeros(X.shape[1], dtype=X.dtype)
            return (safe_sparse_dot(X, X.T, dense_output=True), X_mean)
        n_samples = X.shape[0]
        sample_weight_matrix = sparse.dia_matrix((sqrt_sw, 0), shape=(n_samples, n_samples))
        X_weighted = sample_weight_matrix.dot(X)
        X_mean, _ = mean_variance_axis(X_weighted, axis=0)
        X_mean *= n_samples / sqrt_sw.dot(sqrt_sw)
        X_mX = sqrt_sw[:, None] * safe_sparse_dot(X_mean, X.T, dense_output=True)
        X_mX_m = np.outer(sqrt_sw, sqrt_sw) * np.dot(X_mean, X_mean)
        return (safe_sparse_dot(X, X.T, dense_output=True) + X_mX_m - X_mX - X_mX.T, X_mean)
