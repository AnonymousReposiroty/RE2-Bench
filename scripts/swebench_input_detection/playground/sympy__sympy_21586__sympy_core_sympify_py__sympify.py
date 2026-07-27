from sympy.core.sympify import sympify

def test_input(pred_input):
	assert sympify(a = '{2.0, 3}')==sympify(a = pred_input['args']['a']), 'Prediction failed!'