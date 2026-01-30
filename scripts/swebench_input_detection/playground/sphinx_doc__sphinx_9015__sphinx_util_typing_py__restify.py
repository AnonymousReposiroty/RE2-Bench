from sphinx.util.typing import restify

def test_input(pred_input):
	assert restify(cls = {'__module__': 'tests.test_util_typing'})==restify(cls = pred_input['args']['cls']), 'Prediction failed!'