# Problem: sympy@@sympy_core_operations.py@@_matches_commutative_L202
# Module: sympy.core.operations
# Function: _matches_commutative
# Line: 202

from sympy.core.operations import AssocOp
from sympy import sympify

def test_input(pred_input):
    obj_ins = AssocOp(sympify("sin(x*a_)**n_*cos(x*a_)**m_"))
    obj_ins_pred = AssocOp(sympify(pred_input['self']["__repr__"]))
    assert obj_ins._matches_commutative(expr = sympify('sin(x)'), repl_dict = None, old = False)==obj_ins_pred._matches_commutative(expr = sympify(pred_input['args']['expr']), repl_dict = pred_input['args']['repl_dict'], old = pred_input['args']['old']), 'Prediction failed!'
