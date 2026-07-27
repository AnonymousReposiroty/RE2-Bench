# Problem: sympy__sympy-21260@@sympy.core.symbol.py@@__getnewargs_ex__
# Benchmark: Swebench
# Module: sympy.core.symbol
# Function: __getnewargs_ex__

from sympy.core.symbol import Dummy


def test_input(pred_input):
    obj_ins = Dummy()
    obj_ins_pred = Dummy()
    assert obj_ins.__getnewargs_ex__()==obj_ins_pred.__getnewargs_ex__(), 'Prediction failed!'

