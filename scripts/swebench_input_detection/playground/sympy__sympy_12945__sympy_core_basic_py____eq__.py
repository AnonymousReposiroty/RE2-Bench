from sympy.core.basic import Basic


def test_input(pred_input):
	obj_ins = Basic()
	obj_ins_pred = Basic()
	assert obj_ins.__eq__(other = 'I*x')==obj_ins_pred.__eq__(other = pred_input['args']['other']), 'Prediction failed!'