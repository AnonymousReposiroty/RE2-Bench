from sklearn.utils._array_api import device, get_namespace

def _infer_dimension(spectrum, n_samples):
    """Infers the dimension of a dataset with a given spectrum.

    The returned value will be in [1, n_features - 1].
    """
    xp, _ = get_namespace(spectrum)

    ll = xp.empty_like(spectrum)
    ll[0] = -xp.inf  # we don't want to return n_components = 0
    for rank in range(1, spectrum.shape[0]):
        ll[rank] = _assess_dimension(spectrum, rank, n_samples)
    return xp.argmax(ll)
