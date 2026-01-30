# Problem: sympy__sympy-12489@@sympy.combinatorics.permutations.py@@__invert__
# Benchmark: Swebench
# Module: sympy.combinatorics.permutations
# Function: __invert__

from sympy.combinatorics.permutations import Permutation


def test_input(pred_input):
    obj_ins = Permutation()
    obj_ins._array_form = []
    obj_ins._size = 5
    obj_ins_pred = Permutation()
    obj_ins_pred._array_form = pred_input['self']['_array_form']
    obj_ins_pred._size = pred_input['self']['_size']
    assert obj_ins.__invert__()==obj_ins_pred.__invert__(), 'Prediction failed!'
