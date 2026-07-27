# Problem: sympy@@sympy_core_expr.py@@as_terms_L1226
# Module: sympy.core.expr
# Function: as_terms
# Line: 1226

from sympy.core.expr import Expr
from sympy import sympify

def test_input(pred_input):
    obj_ins = Expr(sympify("-w**2 - x**2 + y**2 + z**2"))
    obj_ins_pred = Expr(sympify(pred_input['self']["__repr__"]))
    assert obj_ins.as_terms()==obj_ins_pred.as_terms(), 'Prediction failed!'