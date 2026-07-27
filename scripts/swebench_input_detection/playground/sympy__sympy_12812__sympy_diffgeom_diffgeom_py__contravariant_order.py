from sympy.diffgeom.diffgeom import contravariant_order

def test_input(pred_input):
	assert contravariant_order(expr = {})==contravariant_order(expr = pred_input['args']['expr']), 'Prediction failed!'