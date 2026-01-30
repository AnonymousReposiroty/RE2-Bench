# Problem: sympy@@sympy_multipledispatch_dispatcher.py@@add_L178
# Module: sympy.multipledispatch.dispatcher
# Function: add
# Line: 178

from sympy.multipledispatch.dispatcher import add
from sympy.matrices.utilities import _iszero

def test_input(pred_input):
    obj_ins = "<dispatched SymmetricHandler>"
    obj_ins_pred = pred_input['self']
    assert obj_ins.add(signature = ["<class 'sympy.matrices.expressions.matadd.MatAdd'>"], func = '<function _ at 0x755441332020>', on_ambiguity = '<function ambiguity_warn at 0x7554489807c0>')==obj_ins_pred.add(signature = pred_input['args']['signature'], func = pred_input['args']['func'], on_ambiguity = pred_input['args']['on_ambiguity']), 'Prediction failed!'