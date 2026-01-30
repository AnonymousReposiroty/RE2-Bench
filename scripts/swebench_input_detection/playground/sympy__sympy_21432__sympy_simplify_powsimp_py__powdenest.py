from sympy.simplify.powsimp import powdenest

def test_input(pred_input):
	assert powdenest(eq = '_x', force = False, polar = False)==powdenest(eq = pred_input['args']['eq'], force = pred_input['kwargs']['force'], polar = pred_input['kwargs']['polar']), 'Prediction failed!'