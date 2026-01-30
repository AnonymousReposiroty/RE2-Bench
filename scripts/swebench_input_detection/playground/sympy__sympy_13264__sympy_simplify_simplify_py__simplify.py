from sympy.simplify.simplify import simplify

def test_input(pred_input):
	assert simplify(expr = '1.0e-14*I')==simplify(expr = pred_input['args']['expr']), 'Prediction failed!'