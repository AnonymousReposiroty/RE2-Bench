# Problem: matplotlib@@matplotlib_colorizer.py@@_ensure_multivariate_data_L837
# Module: matplotlib.colorizer
# Function: _ensure_multivariate_data
# Line: 837
import numpy as np
# from matplotlib.colorizer import _ensure_multivariate_data


def _ensure_multivariate_data(data, n_components):
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



def test_input(pred_input):
    s = """masked_array(data=[1., 2.],
                mask=False,
        fill_value=1e+20)"""
    expr = s.replace("masked_array", "np.ma.masked_array", 1)
    arr = eval(expr, {"np": np})
    assert _ensure_multivariate_data(data = arr, n_components = 1)==_ensure_multivariate_data(data = pred_input['args']['data'], n_components = pred_input['args']['n_components']), 'Prediction failed!'
    
