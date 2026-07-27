# Problem: sympy@@sympy_core_operations.py@@_eval_evalf_L387
# Module: sympy.core.operations
# Function: _eval_evalf
# Line: 387

from sympy.core.operations import AssocOp
from sympy import sqrt
from sympy import sympify
def test_input(pred_input):
    obj_ins = AssocOp(-sqrt(30)/30)
    obj_ins_pred = pred_input['self']
    assert obj_ins._eval_evalf(prec = 2)==obj_ins_pred._eval_evalf(prec = pred_input['args']['prec']), 'Prediction failed!'
