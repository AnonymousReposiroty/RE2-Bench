# Problem: sympy__sympy-23141@@sympy.integrals.intpoly.py@@_polynomial_integrate
# Benchmark: Swebench
# Module: sympy.integrals.intpoly
# Function: _polynomial_integrate

from sympy.integrals.intpoly import _polynomial_integrate
from sympy import sympify

def test_input(pred_input):
    assert _polynomial_integrate(polynomials = {'2': 'x*y'}, facets = None, hp_params = None)==_polynomial_integrate(polynomials = pred_input['args']['polynomials'], facets = pred_input['args']['facets'], hp_params = pred_input['args']['hp_params']), 'Prediction failed!'
    