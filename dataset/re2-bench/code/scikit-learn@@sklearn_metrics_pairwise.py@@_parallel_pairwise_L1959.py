from joblib import effective_n_jobs
from sklearn.utils import check_array, gen_batches, gen_even_slices
from sklearn.utils._array_api import (
    _fill_diagonal,
    _find_matching_floating_dtype,
    _is_numpy_namespace,
    _max_precision_float_dtype,
    _modify_in_place_if_numpy,
    get_namespace,
    get_namespace_and_device,
)
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import _num_samples, check_non_negative

def _parallel_pairwise(X, Y, func, n_jobs, **kwds):
    """Break the pairwise matrix in n_jobs even slices
    and compute them using multithreading."""
    xp, _, device = get_namespace_and_device(X, Y)
    X, Y, dtype_float = _find_floating_dtype_allow_sparse(X, Y, xp=xp)

    if Y is None:
        Y = X

    if effective_n_jobs(n_jobs) == 1:
        return func(X, Y, **kwds)

    # enforce a threading backend to prevent data communication overhead
    fd = delayed(_transposed_dist_wrapper)
    # Transpose `ret` such that a given thread writes its output to a contiguous chunk.
    # Note `order` (i.e. F/C-contiguous) is not included in array API standard, see
    # https://github.com/data-apis/array-api/issues/571 for details.
    # We assume that currently (April 2025) all array API compatible namespaces
    # allocate 2D arrays using the C-contiguity convention by default.
    ret = xp.empty((X.shape[0], Y.shape[0]), device=device, dtype=dtype_float).T
    Parallel(backend="threading", n_jobs=n_jobs)(
        fd(func, ret, s, X, Y[s, ...], **kwds)
        for s in gen_even_slices(_num_samples(Y), effective_n_jobs(n_jobs))
    )

    if (X is Y or Y is None) and func is euclidean_distances:
        # zeroing diagonal for euclidean norm.
        # TODO: do it also for other norms.
        _fill_diagonal(ret, 0, xp=xp)

    # Transform output back
    return ret.T
