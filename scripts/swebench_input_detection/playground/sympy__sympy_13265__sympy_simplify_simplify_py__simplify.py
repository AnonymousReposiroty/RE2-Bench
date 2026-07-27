from sympy.simplify.simplify import simplify

def test_input(pred_input):
	assert simplify(expr = '50')==simplify(expr = pred_input['args']['expr']), 'Prediction failed!'
