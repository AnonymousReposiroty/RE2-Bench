from django.contrib.admindocs.utils import replace_named_groups

def test_input(pred_input):
	assert replace_named_groups(pattern = 'admin/password_change/done/')==replace_named_groups(pattern = pred_input['args']['pattern']), 'Prediction failed!'