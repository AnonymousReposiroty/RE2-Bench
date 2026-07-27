import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement

def _sparse_random_matrix(n_components, n_features, density="auto", random_state=None):
    """Generalized Achlioptas random sparse matrix for random projection.

    Setting density to 1 / 3 will yield the original matrix by Dimitris
    Achlioptas while setting a lower value will yield the generalization
    by Ping Li et al.

    If we note :math:`s = 1 / density`, the components of the random matrix are
    drawn from:

      - -sqrt(s) / sqrt(n_components)   with probability 1 / 2s
      -  0                              with probability 1 - 1 / s
      - +sqrt(s) / sqrt(n_components)   with probability 1 / 2s

    Read more in the :ref:`User Guide <sparse_random_matrix>`.

    Parameters
    ----------
    n_components : int,
        Dimensionality of the target projection space.

    n_features : int,
        Dimensionality of the original source space.

    density : float or 'auto', default='auto'
        Ratio of non-zero component in the random projection matrix in the
        range `(0, 1]`

        If density = 'auto', the value is set to the minimum density
        as recommended by Ping Li et al.: 1 / sqrt(n_features).

        Use density = 1 / 3.0 if you want to reproduce the results from
        Achlioptas, 2001.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generator used to generate the matrix
        at fit time.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    components : {ndarray, sparse matrix} of shape (n_components, n_features)
        The generated Gaussian random matrix. Sparse matrix will be of CSR
        format.

    See Also
    --------
    SparseRandomProjection

    References
    ----------

    .. [1] Ping Li, T. Hastie and K. W. Church, 2006,
           "Very Sparse Random Projections".
           https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf

    .. [2] D. Achlioptas, 2001, "Database-friendly random projections",
           https://cgi.di.uoa.gr/~optas/papers/jl.pdf

    """
    _check_input_size(n_components, n_features)
    density = _check_density(density, n_features)
    rng = check_random_state(random_state)

    if density == 1:
        # skip index generation if totally dense
        components = rng.binomial(1, 0.5, (n_components, n_features)) * 2 - 1
        return 1 / np.sqrt(n_components) * components

    else:
        # Generate location of non zero elements
        indices = []
        offset = 0
        indptr = [offset]
        for _ in range(n_components):
            # find the indices of the non-zero components for row i
            n_nonzero_i = rng.binomial(n_features, density)
            indices_i = sample_without_replacement(
                n_features, n_nonzero_i, random_state=rng
            )
            indices.append(indices_i)
            offset += n_nonzero_i
            indptr.append(offset)

        indices = np.concatenate(indices)

        # Among non zero components the probability of the sign is 50%/50%
        data = rng.binomial(1, 0.5, size=np.size(indices)) * 2 - 1

        # build the CSR structure by concatenating the rows
        components = sp.csr_matrix(
            (data, indices, indptr), shape=(n_components, n_features)
        )

        return np.sqrt(1 / density) / np.sqrt(n_components) * components
