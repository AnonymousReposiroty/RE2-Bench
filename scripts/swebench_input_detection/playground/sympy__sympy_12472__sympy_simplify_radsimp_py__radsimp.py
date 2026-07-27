# Problem: sympy__sympy-12472@@sympy.simplify.radsimp.py@@radsimp
# Benchmark: Swebench
# Module: sympy.simplify.radsimp
# Function: radsimp

from sympy.simplify.radsimp import radsimp
from sympy import sympify

def test_input(pred_input):
    assert radsimp(expr = sympify('1/(a - b)'), symbolic = False, max_terms = 1)==radsimp(expr = pred_input['args']['expr'], symbolic = pred_input['kwargs']['symbolic'], max_terms = pred_input['kwargs']['max_terms']), 'Prediction failed!'
