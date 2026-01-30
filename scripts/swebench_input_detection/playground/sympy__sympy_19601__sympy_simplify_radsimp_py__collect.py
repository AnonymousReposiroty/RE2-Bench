# Problem: sympy__sympy-19601@@sympy.simplify.radsimp.py@@collect
# Benchmark: Swebench
# Module: sympy.simplify.radsimp
# Function: collect

from sympy.simplify.radsimp import collect
from sympy import sympify

def test_input(pred_input):
    assert collect(expr = sympify('_Dummy_373'), syms = sympify('x'), func = None, evaluate = True, exact = False, distribute_order_term = True)==collect(expr = sympify(pred_input['args']['expr']), syms = pred_input['args']['syms'], func = pred_input['args']['func'], evaluate = pred_input['args']['evaluate'], exact = pred_input['args']['exact'], distribute_order_term = pred_input['args']['distribute_order_term']), 'Prediction failed!'
    