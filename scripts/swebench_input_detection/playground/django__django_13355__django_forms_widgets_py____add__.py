from django.forms.widgets import Media

def test_input(pred_input):
	obj_ins = Media()
	obj_ins_pred = Media()
	assert obj_ins.__add__(other = {'_css_lists': {}, '_js_lists': {}})==obj_ins_pred.__add__(other = pred_input['args']['other']), 'Prediction failed!'