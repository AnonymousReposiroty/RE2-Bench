# Problem: sympy__sympy-13962@@sympy.printing.str.py@@sstr
# Benchmark: Swebench
# Module: sympy.printing.str
# Function: sstr

from sympy.printing.str import sstr


def test_input(pred_input):
    assert sstr(expr = {'_array_form': None, '_size': 6}, order = None)==sstr(expr = pred_input['args']['expr'], order = pred_input['kwargs']['order']), 'Prediction failed!'