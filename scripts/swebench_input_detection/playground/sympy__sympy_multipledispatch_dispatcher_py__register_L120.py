# Problem: sympy@@sympy_multipledispatch_dispatcher.py@@register_L120
# Module: sympy.multipledispatch.dispatcher
# Function: register
# Line: 120

from sympy.multipledispatch.dispatcher import register


def test_input(pred_input):
    obj_ins = "<dispatched SymmetricHandler>"
    obj_ins_pred = pred_input['self']
    assert obj_ins.register(types = ["<class 'sympy.matrices.expressions.matpow.MatPow'>"])==obj_ins_pred.register(types = pred_input['args']['types']), 'Prediction failed!'