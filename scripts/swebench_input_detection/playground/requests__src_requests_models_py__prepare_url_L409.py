# Problem: requests@@src_requests_models.py@@prepare_url_L409
# Module: requests.models
# Function: prepare_url
# Line: 409

from requests.models import PreparedRequest


def test_input(pred_input):
    obj_ins = PreparedRequest()
    obj_ins.method = 'POST'
    obj_ins.url = None
    obj_ins.headers = None
    obj_ins._cookies = None
    obj_ins.body = None
    obj_ins.hooks = {'response': []}
    obj_ins._body_position = None
    obj_ins_pred = PreparedRequest()
    obj_ins_pred.method = pred_input['self']['method']
    obj_ins_pred.url = pred_input['self']['url']
    obj_ins_pred.headers = pred_input['self']['headers']
    obj_ins_pred._cookies = pred_input['self']['_cookies']
    obj_ins_pred.body = pred_input['self']['body']
    obj_ins_pred.hooks = pred_input['self']['hooks']
    obj_ins_pred._body_position = pred_input['self']['_body_position']
    assert obj_ins.prepare_url(url = 'http://localhost:49515/', params = {})==obj_ins_pred.prepare_url(url = pred_input['args']['url'], params = pred_input['args']['params']), 'Prediction failed!'