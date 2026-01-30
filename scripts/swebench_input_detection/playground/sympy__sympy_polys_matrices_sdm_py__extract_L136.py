# Problem: sympy@@sympy_polys_matrices_sdm.py@@extract_L136
# Module: sympy.polys.matrices.sdm
# Function: extract
# Line: 136

from sympy.polys.matrices.sdm import SDM
from sympy import ZZ


def test_input(pred_input):
    obj_ins = SDM(shape = [3, 3], domain = ZZ, elemsdict = {})
    obj_ins.rows = 3
    obj_ins.cols = 3
    obj_ins_pred = SDM(shape = pred_input['self']['shape'], domain = ZZ, elemsdict = {})
    obj_ins_pred.rows = pred_input['self']['rows']
    obj_ins_pred.cols = pred_input['self']['cols']
    assert obj_ins.extract(rows = [0, 1, 2], cols = [0])==obj_ins_pred.extract(rows = pred_input['args']['rows'], cols = pred_input['args']['cols']), 'Prediction failed!'
    