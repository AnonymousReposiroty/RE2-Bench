# Problem: sympy__sympy-15599@@sympy.core.mod.py@@doit
# Benchmark: Swebench
# Module: sympy.core.mod
# Function: doit

from sympy.core.mod import Mod
from sympy import sympify

def test_input(pred_input):
    obj_ins = Mod(sympify("2*t"), sympify("t"))
    obj_ins_pred = Mod(sympify(pred_input['args']['p']), sympify(pred_input['args']['q']))
    assert obj_ins.doit(p = '2*t', q = 't')==obj_ins_pred.doit(p = pred_input['args']['p'], q = pred_input['args']['q']), 'Prediction failed!'
