# Problem: sympy@@sympy.matrices.reductions.py@@_rank_L220
# Module: sympy.matrices.reductions
# Function: _rank
# Line: 220

from sympy import Matrix
from sympy.matrices.reductions import _rank
from sympy.matrices.utilities import _iszero


def parse_matrix(s):
    """Parse a string representation of a Matrix."""
    # s is like "Matrix([[1, 1, 1],[1, 1, 1],[1, 1, 1]])"
    # Use eval with Matrix in the namespace
    return eval(s, {'Matrix': Matrix})


def test_input(pred_input):
    # Ground truth: a 3x3 matrix of all ones
    M_gt = Matrix([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])

    # Parse predicted Matrix from string
    M_pred = parse_matrix(pred_input['args']['M'])

    # Get simplify parameter
    simplify_gt = False
    simplify_pred = pred_input['args']['simplify']

    # Call _rank on both
    result_gt = _rank(M=M_gt, iszerofunc=_iszero, simplify=simplify_gt)
    result_pred = _rank(M=M_pred, iszerofunc=_iszero, simplify=simplify_pred)

    # Compare results
    assert result_gt == result_pred, 'Prediction failed!'
