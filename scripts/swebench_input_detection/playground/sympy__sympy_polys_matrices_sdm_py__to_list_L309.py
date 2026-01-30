# Problem: sympy@@sympy_polys_matrices_sdm.py@@to_list_L309
# Module: sympy.polys.matrices.sdm
# Function: to_list
# Line: 309
from sympy import Matrix
from sympy.polys.matrices.sdm import SDM


def test_input(pred_input):
    obj_ins = SDM()
    obj_ins_pred = SDM()
    assert obj_ins.to_list(M = {'0': {'0': Matrix([[4, 2],[2, 3],[7, 5]]), '1': Matrix([[1, 1, 1],[1, 1, 1],[1, 1, 1]])}, '1': {'0': Matrix([[1, 0],[0, 1]]), '1': Matrix([[1, 2, 3],[3, 5, 4]])}})==obj_ins_pred.to_list(M = pred_input['args']['M']), 'Prediction failed!'
