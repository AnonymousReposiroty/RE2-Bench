# Problem: matplotlib@@matplotlib_colorizer.py@@_ensure_multivariate_data_L837
# Module: matplotlib.colorizer
# Function: _ensure_multivariate_data
# Line: 837

import numpy as np

# Copy of the function from matplotlib/colorizer.py (line 837-909)
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
    """
    if isinstance(data, np.ndarray):
        if len(data.dtype.descr) == n_components:
            # pass scalar data and already formatted data
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
        # convert data from shape (n_components, n, m) to (n, m) with a new dtype
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


def reconstruct_masked_array(data_dict):
    """
    Reconstruct a numpy masked array from its dictionary representation.
    """
    data = np.array(data_dict['data'])
    mask = data_dict['mask']
    fill_value = data_dict.get('fill_value', None)

    return np.ma.array(data, mask=mask, fill_value=fill_value)


def test_input(pred_input):
    """
    Test case for _ensure_multivariate_data function.

    Ground truth input:
    - data: masked array with data=[1., 2.], mask=False, fill_value=1e+20
    - n_components: 1

    Expected: returns data unchanged (since n_components == 1)
    """
    # Create ground truth data
    data_gt = np.ma.array([1., 2.], mask=False, fill_value=1e+20)
    n_components_gt = 1

    # Call with ground truth input
    result_gt = _ensure_multivariate_data(data_gt, n_components_gt)

    # Create predicted data from input
    data_dict = pred_input['args']['data']
    data_pred = reconstruct_masked_array(data_dict)
    n_components_pred = data_dict.get('n_components', 1)

    # Call with predicted input
    result_pred = _ensure_multivariate_data(data_pred, n_components_pred)

    # Compare results
    if isinstance(result_gt, np.ndarray):
        assert np.array_equal(np.ma.getdata(result_gt), np.ma.getdata(result_pred)), \
            f'Prediction failed! Data mismatch'
        assert np.array_equal(np.ma.getmask(result_gt), np.ma.getmask(result_pred)), \
            f'Prediction failed! Mask mismatch'
    else:
        assert result_gt == result_pred, f'Prediction failed! Expected {result_gt}, got {result_pred}'


def test_output(pred_output):
    """
    Test case for _ensure_multivariate_data function output prediction.

    Ground truth:
    - data: masked array with data=[1., 2.], mask=False, fill_value=1e+20
    - n_components: 1
    - return: the same masked array (unchanged for n_components=1)
    """
    # Create ground truth data
    data_gt = np.ma.array([1., 2.], mask=False, fill_value=1e+20)
    n_components_gt = 1

    # Call with ground truth input
    result_gt = _ensure_multivariate_data(data_gt, n_components_gt)

    # Compare with prediction
    if isinstance(result_gt, np.ndarray):
        pred_data = np.array(pred_output) if not isinstance(pred_output, np.ndarray) else pred_output
        assert np.array_equal(np.ma.getdata(result_gt), np.ma.getdata(pred_data)), \
            f'Prediction failed! Expected {result_gt}, got {pred_output}'
    else:
        assert result_gt == pred_output, f'Prediction failed! Expected {result_gt}, got {pred_output}'


if __name__ == "__main__":
    # Test with ground truth input
    gt_input = {
        'self': {},
        'args': {
            'data': {
                'data': [1., 2.],
                'mask': False,
                'fill_value': 1e+20,
                'n_components': 1
            }
        },
        'kwargs': {}
    }

    test_input(gt_input)
    print("Input test passed!")

    # For output test, get the actual result first
    data_gt = np.ma.array([1., 2.], mask=False, fill_value=1e+20)
    gt_output = _ensure_multivariate_data(data_gt, 1)
    print(f"Ground truth output: {gt_output}")
    print(f"Output type: {type(gt_output)}")

    test_output(gt_output)
    print("Output test passed!")
