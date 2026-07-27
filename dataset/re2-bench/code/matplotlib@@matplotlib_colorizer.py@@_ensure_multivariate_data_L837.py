import numpy as np

def _ensure_multivariate_data(data, n_components):
    """
    Ensure that the data has dtype with n_components.
    Input data of shape (n_components, n, m) is converted to an array of shape
    (n, m) with data type np.dtype(f'{data.dtype}, ' * n_components)
    Complex data is returned as a view with dtype np.dtype('float64, float64')
    or np.dtype('float32, float32')
    If n_components is 1 and data is not of type np.ndarray (i.e. PIL.Image),
    the data is returned unchanged.
    If data is None, the function returns None

    Parameters
    ----------
    n_components : int
        Number of variates in the data.
    data : np.ndarray, PIL.Image or None

    Returns
    -------
    np.ndarray, PIL.Image or None
    """

    if isinstance(data, np.ndarray):
        if len(data.dtype.descr) == n_components:
            # pass scalar data
            # and already formatted data
            return data
        elif data.dtype in [np.complex64, np.complex128]:
            if n_components != 2:
                raise ValueError("Invalid data entry for multivariate data. "
                                 "Complex numbers are incompatible with "
                                 f"{n_components} variates.")

            # pass complex data
            if data.dtype == np.complex128:
                dt = np.dtype('float64, float64')
            else:
                dt = np.dtype('float32, float32')

            reconstructed = np.ma.array(np.ma.getdata(data).view(dt))
            if np.ma.is_masked(data):
                for descriptor in dt.descr:
                    reconstructed[descriptor[0]][data.mask] = np.ma.masked
            return reconstructed

    if n_components > 1 and len(data) == n_components:
        # convert data from shape (n_components, n, m)
        # to (n, m) with a new dtype
        data = [np.ma.array(part, copy=False) for part in data]
        dt = np.dtype(', '.join([f'{part.dtype}' for part in data]))
        fields = [descriptor[0] for descriptor in dt.descr]
        reconstructed = np.ma.empty(data[0].shape, dtype=dt)
        for i, f in enumerate(fields):
            if data[i].shape != reconstructed.shape:
                raise ValueError("For multivariate data all variates must have same "
                                 f"shape, not {data[0].shape} and {data[i].shape}")
            reconstructed[f] = data[i]
            if np.ma.is_masked(data[i]):
                reconstructed[f][data[i].mask] = np.ma.masked
        return reconstructed

    if n_components == 1:
        # PIL.Image gets passed here
        return data

    elif n_components == 2:
        raise ValueError("Invalid data entry for multivariate data. The data"
                         " must contain complex numbers, or have a first dimension 2,"
                         " or be of a dtype with 2 fields")
    else:
        raise ValueError("Invalid data entry for multivariate data. The shape"
                         f" of the data must have a first dimension {n_components}"
                         f" or be of a dtype with {n_components} fields")
