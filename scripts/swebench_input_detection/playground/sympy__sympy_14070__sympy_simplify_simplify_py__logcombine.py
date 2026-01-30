from sympy.simplify.simplify import logcombine


def test_input(pred_input):
	assert logcombine(expr = '2', force = False)==logcombine(expr = pred_input['args']['expr'], force = pred_input['kwargs']['force']), 'Prediction failed!'