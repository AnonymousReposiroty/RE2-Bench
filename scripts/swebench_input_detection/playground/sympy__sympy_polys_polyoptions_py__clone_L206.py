# Problem: sympy@@sympy_polys_polyoptions.py@@clone_L206
# Module: sympy.polys.polyoptions
# Function: clone
# Line: 206

from sympy.polys.polyoptions import Options
from sympy import sympify

def test_input(pred_input):
    obj_ins = Options(gens=[sympify('x')], args={})
    obj_ins_pred = Options(gens=[sympify('x')], args={})
    assert obj_ins.clone(updates = {'gens': ['cos(phi)']})==obj_ins_pred.clone(updates = pred_input['args']['updates']), 'Prediction failed!'