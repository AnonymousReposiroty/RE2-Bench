# Problem: sympy@@sympy_core_expr.py@@_eval_is_extended_positive_negative_L912
# Module: sympy.core.expr
# Function: _eval_is_extended_positive_negative
# Line: 912

from sympy.core.expr import Expr
from sympy import sympify

def test_input(pred_input):
    obj_ins = Expr(sympify("-2"))
    obj_ins_pred = Expr(sympify(pred_input['self']["__repr__"]))
    assert obj_ins._eval_is_extended_positive_negative(positive = True)==obj_ins_pred._eval_is_extended_positive_negative(positive = pred_input['args']['positive']), 'Prediction failed!'