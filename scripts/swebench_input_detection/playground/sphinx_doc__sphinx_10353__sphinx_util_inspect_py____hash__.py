from sphinx.util.inspect import TypeAliasForwardRef


def test_input(pred_input):
	obj_ins = TypeAliasForwardRef(name = 'example')
	obj_ins_pred = TypeAliasForwardRef(name = pred_input['self']['name'])
	assert obj_ins.__hash__()==obj_ins_pred.__hash__(), 'Prediction failed!'