from sympy.matrices.expressions.matmul import MatMul



def test_input(pred_input):
	obj_ins = MatMul()
	obj_ins_pred = MatMul()
	assert obj_ins.args_cnc()==obj_ins_pred.args_cnc(), 'Prediction failed!'