# Problem: sympy__sympy-23117@@sympy.tensor.array.ndim_array.py@@f_L195
# Module: sympy.tensor.array.ndim_array
# Function: f (nested function inside _scan_iterable_shape)
# Line: 195

from sympy import symbols, sin, cos, sympify
from sympy.tensor.array.ndim_array import NDimArray


def test_input(pred_input):
    # Ground truth: pointer is a sympy expression sin(x)**2 + cos(x)**2
    # Note: The input has escaped quotes which seems to be a logging artifact
    # The actual expression should be sin(x)**2 + cos(x)**2
    x = symbols('x')
    pointer_gt = sin(x)**2 + cos(x)**2

    # Parse predicted input
    pointer_str = pred_input['args']['pointer']
    # Clean up the string - remove extra quotes if present
    pointer_str_clean = pointer_str.replace('"', '')
    pointer_pred = sympify(pointer_str_clean)

    # f is a nested function inside _scan_iterable_shape, so we test through the parent
    # _scan_iterable_shape calls f internally
    # For a single sympy expression (not iterable), it should return ([expr], ())
    result_gt = NDimArray._scan_iterable_shape(pointer_gt)
    result_pred = NDimArray._scan_iterable_shape(pointer_pred)

    # Compare results
    assert result_gt == result_pred, f'Prediction failed! Expected: {result_gt}, Got: {result_pred}'


if __name__ == "__main__":
    # Test with the given input
    test_input_data = {
        "self": {},
        "args": {
            "pointer": "sin(x)**2 + \"cos(x)\"**2"
        },
        "kwargs": {}
    }

    test_input(test_input_data)
    print("Test passed!")
