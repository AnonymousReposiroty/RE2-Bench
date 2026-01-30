# Problem: sympy@@sympy_polys_rings.py@@cancel_L3202
# Module: sympy.polys.rings
# Function: cancel
# Line: 3202

from sympy.polys.rings import PolyElement


def test_input(pred_input):
    obj_ins = PolyElement(ring = 'Polynomial ring in x, y, z, w over ZZ with lex order')
    obj_ins_pred = PolyElement(ring = pred_input['self']['ring'])
    assert obj_ins.cancel(g = {'(0, 0, 0, 0)': 1})==obj_ins_pred.cancel(g = pred_input['args']['g']), 'Prediction failed!'