from . import _api, _version, cbook, _docstring, rcsetup

def get_backend(*, auto_select=True):
    """
    Return the name of the current backend.

    Parameters
    ----------
    auto_select : bool, default: True
        Whether to trigger backend resolution if no backend has been
        selected so far. If True, this ensures that a valid backend
        is returned. If False, this returns None if no backend has been
        selected so far.

        .. versionadded:: 3.10

        .. admonition:: Provisional

           The *auto_select* flag is provisional. It may be changed or removed
           without prior warning.

    See Also
    --------
    matplotlib.use
    """
    if auto_select:
        return rcParams['backend']
    else:
        backend = rcParams._get('backend')
        if backend is rcsetup._auto_backend_sentinel:
            return None
        else:
            return backend
