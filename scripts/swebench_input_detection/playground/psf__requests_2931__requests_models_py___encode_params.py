from requests.models import RequestEncodingMixin

def test_input(pred_input):
	obj_ins = RequestEncodingMixin()
	obj_ins_pred = RequestEncodingMixin()
	assert obj_ins._encode_params(data = {})==obj_ins_pred._encode_params(data = pred_input['args']['data']), 'Prediction failed!'
