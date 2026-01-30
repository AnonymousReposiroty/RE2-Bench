# Problem: sympy@@sympy_core_numbers.py@@__lt___L1999
# Module: sympy.core.numbers
# Function: __lt__
# Line: 1999

from sympy.core.numbers import Number


def test_input(pred_input):
    obj_ins = Number(3)
    obj_ins_pred = Number(pred_input['self']["__repr__"])
    assert obj_ins.__lt__(other = Number(1))==obj_ins_pred.__lt__(other = Number(pred_input['args']['other'])), 'Prediction failed!'
