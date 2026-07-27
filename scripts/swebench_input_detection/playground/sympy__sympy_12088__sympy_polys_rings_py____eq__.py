# Problem: sympy__sympy-12088@@sympy.polys.rings.py@@__eq__
# Benchmark: Swebench
# Module: sympy.polys.rings
# Function: __eq__

from sympy.polys.rings import PolyElement


def test_input(pred_input):
    obj_ins = PolyElement()
    obj_ins_pred = PolyElement()
    assert obj_ins.__eq__(p1 = {'_hash': 1651261762238379522}, p2 = 1)==obj_ins_pred.__eq__(p1 = pred_input['args']['p1'], p2 = pred_input['args']['p2']), 'Prediction failed!'

