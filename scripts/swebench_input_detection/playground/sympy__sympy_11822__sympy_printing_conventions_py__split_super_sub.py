from sympy.printing.conventions import split_super_sub

def test_input(pred_input):
	assert split_super_sub(text = 'alpha^+_1')==split_super_sub(text = pred_input['args']['text']), 'Prediction failed!'