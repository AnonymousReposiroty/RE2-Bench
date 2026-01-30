# Problem: sympy__sympy-18189@@sympy.solvers.diophantine.py@@diophantine
# Benchmark: Swebench
# Module: sympy.solvers.diophantine
# Function: diophantine

from sympy.solvers.diophantine import diophantine
from sympy import sympify

def test_input(pred_input):
    assert diophantine(eq = sympify('8*x*y + z**2'))==diophantine(eq = sympify(pred_input['args']['eq'])), 'Prediction failed!'
