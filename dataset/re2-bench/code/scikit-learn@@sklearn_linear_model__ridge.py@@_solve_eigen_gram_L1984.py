import numpy as np
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

    @staticmethod
    def _decomp_diag(v_prime, Q):
        xp, _ = get_namespace(v_prime, Q)
        return xp.sum(v_prime * Q ** 2, axis=1)

    @staticmethod
    def _diag_dot(D, B):
        xp, _ = get_namespace(D, B)
        if len(B.shape) > 1:
            D = D[(slice(None),) + (None,) * (len(B.shape) - 1)]
        return D * B

    def _solve_eigen_gram(self, alpha, y, sqrt_sw, X_mean, eigvals, Q, QT_y):
        """Compute dual coefficients and diagonal of G^-1.

        Used when we have a decomposition of X.X^T (n_samples <= n_features).
        """
        xp, is_array_api = get_namespace(eigvals)
        w = 1.0 / (eigvals + alpha)
        if self.fit_intercept:
            norm = xp.linalg.vector_norm if is_array_api else np.linalg.norm
            normalized_sw = sqrt_sw / norm(sqrt_sw)
            intercept_dim = _find_smallest_angle(normalized_sw, Q)
            w[intercept_dim] = 0
        c = Q @ self._diag_dot(w, QT_y)
        G_inverse_diag = self._decomp_diag(w, Q)
        if len(y.shape) != 1:
            G_inverse_diag = G_inverse_diag[:, None]
        return (G_inverse_diag, c)
