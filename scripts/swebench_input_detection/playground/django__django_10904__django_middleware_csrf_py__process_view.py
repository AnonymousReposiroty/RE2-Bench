from django.middleware.csrf import CsrfViewMiddleware
obj = CsrfViewMiddleware(get_response={})


def test_input(pred_input):
	obj_ins = CsrfViewMiddleware(get_response = None)
	obj_ins_pred = CsrfViewMiddleware(get_response = pred_input['self']['get_response'])
	assert obj_ins.process_view(request = {'GET': {'_encoding': 'utf-8', '_mutable': True}, '_post': {'_encoding': 'utf-8', '_mutable': True}, 'COOKIES': {'csrftoken': 'ABC1bcdefghij2bcdefghij3bcdefghij4bcdefghij5bcdefghij6bcdefghijA'}, 'META': {'CSRF_COOKIE': 'ABC1bcdefghij2bcdefghij3bcdefghij4bcdefghij5bcdefghij6bcdefghijA'}, 'FILES': {}, 'path': '', 'path_info': '', 'method': 'POST', 'resolver_match': None, 'content_type': None, 'content_params': None, 'raise_error': True, 'session': {'_csrftoken': 'ABC1bcdefghij2bcdefghij3bcdefghij4bcdefghij5bcdefghij6bcdefghijA'}}, callback = {}, callback_args = [], callback_kwargs = {})==obj_ins_pred.process_view(request = pred_input['args']['request'], callback = pred_input['args']['callback'], callback_args = pred_input['args']['callback_args'], callback_kwargs = pred_input['args']['callback_kwargs']), 'Prediction failed!'