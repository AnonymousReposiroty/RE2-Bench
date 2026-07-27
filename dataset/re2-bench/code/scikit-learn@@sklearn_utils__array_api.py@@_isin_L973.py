import numpy

def _isin(element, test_elements, xp, assume_unique=False, invert=False):
    """Calculates ``element in test_elements``, broadcasting over `element`
    only.

    Returns a boolean array of the same shape as `element` that is True
    where an element of `element` is in `test_elements` and False otherwise.
    """
    if _is_numpy_namespace(xp):
        return xp.asarray(
            numpy.isin(
                element=element,
                test_elements=test_elements,
                assume_unique=assume_unique,
                invert=invert,
            )
        )

    original_element_shape = element.shape
    element = xp.reshape(element, (-1,))
    test_elements = xp.reshape(test_elements, (-1,))
    return xp.reshape(
        _in1d(
            ar1=element,
            ar2=test_elements,
            xp=xp,
            assume_unique=assume_unique,
            invert=invert,
        ),
        original_element_shape,
    )
