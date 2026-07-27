import numpy

def _ravel(array, xp=None):
    """Array API compliant version of np.ravel.

    For non numpy namespaces, it just returns a flattened array, that might
    be or not be a copy.
    """
    xp, _ = get_namespace(array, xp=xp)
    if _is_numpy_namespace(xp):
        array = numpy.asarray(array)
        return xp.asarray(numpy.ravel(array, order="C"))

    return xp.reshape(array, shape=(-1,))
