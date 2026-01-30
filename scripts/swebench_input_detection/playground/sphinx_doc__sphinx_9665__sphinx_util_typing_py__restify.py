from sphinx.util.typing import restify

def test_input(pred_input):
	assert restify(cls = None)==restify(cls = pred_input['args']['cls']), 'Prediction failed!'