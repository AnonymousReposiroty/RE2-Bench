# Problem: sympy@@sympy_core_expr.py@@as_content_primitive_L2171
# Module: sympy.core.expr
# Function: as_content_primitive
# Line: 2171

from sympy.core.expr import Expr


def test_input(pred_input):
    obj_ins = Expr()
    obj_ins_pred = Expr()
    assert obj_ins.as_content_primitive(radical = False, clear = False)==obj_ins_pred.as_content_primitive(radical = pred_input['args']['radical'], clear = pred_input['args']['clear']), 'Prediction failed!'
    