from django.contrib.admindocs.utils import replace_unnamed_groups

def test_input(pred_input):
	assert replace_unnamed_groups(pattern = '^a/?$')==replace_unnamed_groups(pattern = pred_input['args']['pattern']), 'Prediction failed!'