from sphinx.util.typing import stringify

def test_input(pred_input):
	assert stringify({'__module__': 'tests.test_util_typing', '__doc__': None})==stringify(pred_input['args']['annotation']), 'Prediction failed!'