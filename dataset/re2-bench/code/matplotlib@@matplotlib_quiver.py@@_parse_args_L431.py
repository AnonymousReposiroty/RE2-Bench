import numpy as np
from matplotlib import _api, cbook, _docstring

def _parse_args(*args, caller_name='function'):
    """
    Helper function to parse positional parameters for colored vector plots.

    This is currently used for Quiver and Barbs.

    Parameters
    ----------
    *args : list
        list of 2-5 arguments. Depending on their number they are parsed to::

            U, V
            U, V, C
            X, Y, U, V
            X, Y, U, V, C

    caller_name : str
        Name of the calling method (used in error messages).
    """
    X = Y = C = None

    nargs = len(args)
    if nargs == 2:
        # The use of atleast_1d allows for handling scalar arguments while also
        # keeping masked arrays
        U, V = np.atleast_1d(*args)
    elif nargs == 3:
        U, V, C = np.atleast_1d(*args)
    elif nargs == 4:
        X, Y, U, V = np.atleast_1d(*args)
    elif nargs == 5:
        X, Y, U, V, C = np.atleast_1d(*args)
    else:
        raise _api.nargs_error(caller_name, takes="from 2 to 5", given=nargs)

    nr, nc = (1, U.shape[0]) if U.ndim == 1 else U.shape

    if X is not None:
        X = X.ravel()
        Y = Y.ravel()
        if len(X) == nc and len(Y) == nr:
            X, Y = (a.ravel() for a in np.meshgrid(X, Y))
        elif len(X) != len(Y):
            raise ValueError('X and Y must be the same size, but '
                             f'X.size is {X.size} and Y.size is {Y.size}.')
    else:
        indexgrid = np.meshgrid(np.arange(nc), np.arange(nr))
        X, Y = (np.ravel(a) for a in indexgrid)
    # Size validation for U, V, C is left to the set_UVC method.
    return X, Y, U, V, C
