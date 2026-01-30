# Problem: sympy__sympy-18033@@sympy.combinatorics.permutations.py@@__pow__
# Benchmark: Swebench
# Module: sympy.combinatorics.permutations
# Function: __pow__

from sympy.combinatorics.permutations import Permutation


def test_input(pred_input):
    obj_ins = Permutation()
    obj_ins._array_form = []
    obj_ins._size = 7
    obj_ins._cyclic_form = []
    obj_ins_pred = Permutation()
    obj_ins_pred._array_form = pred_input['self']['_array_form']
    obj_ins_pred._size = pred_input['self']['_size']
    obj_ins_pred._cyclic_form = pred_input['self']['_cyclic_form']
    assert obj_ins.__pow__(n = -2)==obj_ins_pred.__pow__(n = pred_input['args']['n']), 'Prediction failed!'
    

