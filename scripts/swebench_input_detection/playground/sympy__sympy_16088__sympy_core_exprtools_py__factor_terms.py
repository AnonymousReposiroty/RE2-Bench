from sympy.core.exprtools import factor_terms

def test_input(pred_input):
	assert factor_terms(expr = 'R3 + R4 + 4050')==factor_terms(expr = pred_input['args']['expr']), 'Prediction failed!'