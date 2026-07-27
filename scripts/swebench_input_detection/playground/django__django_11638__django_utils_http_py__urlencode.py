from django.utils.http import urlencode

def test_input(pred_input):
	assert urlencode(query = [['a', 1], ['b', 2], ['c', 3]])==urlencode(query = pred_input['args']['query']), 'Prediction failed!'