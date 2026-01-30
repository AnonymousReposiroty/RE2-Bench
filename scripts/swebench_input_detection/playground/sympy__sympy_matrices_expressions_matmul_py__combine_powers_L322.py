# Problem: sympy@@sympy.matrices.expressions.matmul.py@@combine_powers_L322
# Module: sympy.matrices.expressions.matmul
# Function: combine_powers
# Line: 322

from sympy import symbols
from sympy.matrices import MatrixSymbol
from sympy.matrices.expressions.matmul import MatMul, combine_powers


# Define matrix symbols with a common shape
n = symbols('n', integer=True)
X = MatrixSymbol('X', n, n)
Z = MatrixSymbol('Z', n, n)


def parse_matmul(expr_str):
    """Parse a string representation of matrix multiplication."""
    # Create a local namespace with our matrix symbols
    local_dict = {'X': X, 'Z': Z, 'MatMul': MatMul}
    # Parse the expression
    # Handle "X*Z" format
    if '*' in expr_str:
        parts = expr_str.split('*')
        matrices = [local_dict[p.strip()] for p in parts]
        return MatMul(*matrices)
    return local_dict.get(expr_str.strip())


def test_input(pred_input):
    # Ground truth: X*Z as a MatMul
    mul_gt = MatMul(X, Z)

    # Parse predicted input
    mul_pred = parse_matmul(pred_input['args']['mul'])

    # Call combine_powers on both
    result_gt = combine_powers(mul_gt)
    result_pred = combine_powers(mul_pred)

    # Compare results
    assert result_gt == result_pred, 'Prediction failed!'
