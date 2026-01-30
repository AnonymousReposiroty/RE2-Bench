# Problem: sympy@@sympy_polys_matrices_domainmatrix.py@@to_list_L880
# Module: sympy.polys.matrices.domainmatrix
# Function: to_list
# Line: 880
from sympy import Matrix
from sympy.polys.matrices.domainmatrix import DomainMatrix


def test_input(pred_input):
    obj_ins = DomainMatrix()
    obj_ins.rep = {'0': {'0': Matrix([[1, 0],[0, 1]]), '1': Matrix([[1, 2, 3],[3, 5, 4]])}, '1': {'0': Matrix([[4, 2],[2, 3],[7, 5]]), '1': Matrix([[1, 1, 1],[1, 1, 1],[1, 1, 1]])}}
    obj_ins.shape = [2, 2]
    obj_ins.domain = 'EXRAW'
    obj_ins_pred = DomainMatrix()
    obj_ins_pred.rep = pred_input['self']['rep']
    obj_ins_pred.shape = pred_input['self']['shape']
    obj_ins_pred.domain = pred_input['self']['domain']
    assert obj_ins.to_list()==obj_ins_pred.to_list(), 'Prediction failed!'
    
