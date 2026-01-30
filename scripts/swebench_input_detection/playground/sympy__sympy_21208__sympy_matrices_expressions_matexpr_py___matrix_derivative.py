# Problem: sympy__sympy-21208@@sympy.matrices.expressions.matexpr.py@@_matrix_derivative_L648
# Module: sympy.matrices.expressions.matexpr
# Function: _matrix_derivative
# Line: 648

import sys

from sympy import symbols, MatrixSymbol
from sympy.matrices.expressions.matexpr import _matrix_derivative


def dict_to_matrix_expr(d, name="X"):
    """Convert dict representation to MatrixSymbol."""
    rows = d.get("_rows")
    cols = d.get("_cols")
    mat = d.get("_mat")
    if mat is None:
        # Create a MatrixSymbol
        return MatrixSymbol(name, rows, cols)
    else:
        from sympy import ImmutableDenseMatrix
        return ImmutableDenseMatrix(mat)


def test_input(pred_input):
    # Ground truth: expr is a 2x2 MatrixSymbol, x is symbol 'x'
    # Note: _matrix_derivative expects x to have a shape attribute for matrix expressions
    # So we create x as a MatrixSymbol (column vector)
    x_gt = MatrixSymbol('x', 2, 1)
    expr_gt = MatrixSymbol('X', 2, 2)

    # Parse predicted inputs
    expr_dict = pred_input['args']['expr']
    x_str = pred_input['args']['x']

    expr_pred = dict_to_matrix_expr(expr_dict, name="X")
    # Create x as a MatrixSymbol with compatible dimensions
    x_pred = MatrixSymbol(x_str, expr_pred.shape[0], 1)

    # Call _matrix_derivative on both
    result_gt = _matrix_derivative(expr_gt, x_gt)
    result_pred = _matrix_derivative(expr_pred, x_pred)

    # Compare results
    assert result_gt == result_pred, f'Prediction failed! Expected: {result_gt}, Got: {result_pred}'


if __name__ == "__main__":
    # Test with the given input
    test_input_data = {
        "self": {},
        "args": {
            "expr": {
                "_rows": 2,
                "_cols": 2,
                "_mat": None
            },
            "x": "x"
        },
        "kwargs": {}
    }

    test_input(test_input_data)
    print("Test passed!")
