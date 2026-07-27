# Problem: sympy@@sympy_matrices_expressions_blockmatrix.py@@__new___L80
# Module: sympy.matrices.expressions.blockmatrix
# Function: __new__
# Line: 80

from sympy.matrices.expressions.blockmatrix import BlockMatrix

from sympy import Matrix, sympify

def convert_matrix_string(obj):
    """
    Recursively convert any string of the form 'Matrix([...])'
    inside nested Python lists into a real SymPy Matrix.
    """
    # Case 1: list -> recurse
    if isinstance(obj, list):
        return [convert_matrix_string(x) for x in obj]
    
    # Case 2: string -> convert if it looks like a Matrix(...)
    if isinstance(obj, str) and obj.strip().startswith("Matrix("):
        return sympify(obj, locals={"Matrix": Matrix})
    
    # Base case: return as-is
    return obj
def test_input(pred_input):
    obj_ins = BlockMatrix()
    obj_ins_pred = BlockMatrix()
    assert obj_ins.__new__(cls = BlockMatrix, args = [[['Matrix([\n[4, 2],\n[2, 3],\n[7, 5]])', 'Matrix([\n[1, 1, 1],\n[1, 1, 1],\n[1, 1, 1]])'], ['Matrix([\n[1, 0],\n[0, 1]])', 'Matrix([\n[1, 2, 3],\n[3, 5, 4]])']]])==obj_ins_pred.__new__(cls = pred_input['args']['cls'], args = pred_input['args']['args']), 'Prediction failed!'
