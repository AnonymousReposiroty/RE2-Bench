# Problem: sympy@@sympy_ntheory_multinomial.py@@multinomial_coefficients_L55
# Module: sympy.ntheory.multinomial
# Function: multinomial_coefficients
# Line: 55

from sympy.ntheory.multinomial import multinomial_coefficients


def test_input(pred_input):
    assert multinomial_coefficients(m = 4, n = 2)==multinomial_coefficients(m = pred_input['args']['m'], n = pred_input['args']['n']), 'Prediction failed!'
