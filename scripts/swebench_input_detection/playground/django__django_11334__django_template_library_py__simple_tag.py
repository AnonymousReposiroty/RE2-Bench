from django.template.library import Library

obj = Library()


def test_input(pred_input):
	obj_ins = Library()
	obj_ins_pred = Library()
	assert obj_ins.simple_tag(func = None)==obj_ins_pred.simple_tag(func = pred_input['args']['func']), 'Prediction failed!'