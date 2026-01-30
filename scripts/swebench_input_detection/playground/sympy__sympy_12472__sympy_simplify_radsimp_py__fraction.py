from sympy.simplify.radsimp import fraction

def test_input(pred_input):
	assert fraction(expr = '-x**3')==fraction(expr = pred_input['args']['expr']), 'Prediction failed!'