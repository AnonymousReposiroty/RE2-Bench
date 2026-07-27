import numpy
from sklearn.externals import array_api_compat

def _median(x, axis=None, keepdims=False, xp=None):
    # XXX: `median` is not included in the array API spec, but is implemented
    # in most array libraries, and all that we support (as of May 2025).
    # TODO: consider simplifying this code to use scipy instead once the oldest
    # supported SciPy version provides `scipy.stats.quantile` with native array API
    # support (likely scipy 1.16 at the time of writing). Proper benchmarking of
    # either option with popular array namespaces is required to evaluate the
    # impact of this choice.
    xp, _, device = get_namespace_and_device(x, xp=xp)

    # `torch.median` takes the lower of the two medians when `x` has even number
    # of elements, thus we use `torch.quantile(q=0.5)`, which gives mean of the two
    if array_api_compat.is_torch_namespace(xp):
        return xp.quantile(x, q=0.5, dim=axis, keepdim=keepdims)

    if hasattr(xp, "median"):
        return xp.median(x, axis=axis, keepdims=keepdims)

    # Intended mostly for array-api-strict (which as no "median", as per the spec)
    # as `_convert_to_numpy` does not necessarily work for all array types.
    x_np = _convert_to_numpy(x, xp=xp)
    return xp.asarray(numpy.median(x_np, axis=axis, keepdims=keepdims), device=device)
