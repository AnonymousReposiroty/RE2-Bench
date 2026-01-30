from sympy.core.expr import Expr

def test_input(pred_input):
	obj_ins = Expr()
	obj_ins_pred = Expr()
	assert obj_ins.__eq__(other = 'picoweber')==obj_ins_pred.__eq__(other = pred_input['args']['other']), 'Prediction failed!'