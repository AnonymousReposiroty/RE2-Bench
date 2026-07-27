# Problem: sympy@@sympy_logic_boolalg.py@@__new___L1328
# Module: sympy.logic.boolalg
# Function: __new__
# Line: 1328

from sympy.logic.boolalg import Equivalent


def test_input(pred_input):
    obj_ins = Equivalent()
    obj_ins_pred = Equivalent()
    assert obj_ins.__new__(cls = 'Equivalent', evaluate = None, args = ['Q.even(X) | Q.even(Z)', 'Q.even(X*Z)'])==obj_ins_pred.__new__(cls = pred_input['args']['cls'], evaluate = pred_input['args']['evaluate'], args = pred_input['args']['args']), 'Prediction failed!'
    
