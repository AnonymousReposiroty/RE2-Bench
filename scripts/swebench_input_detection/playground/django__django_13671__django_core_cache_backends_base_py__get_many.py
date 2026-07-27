from django.core.cache.backends.base import BaseCache

def test_input(pred_input):
	obj_ins = BaseCache(params={'default_timeout': 300, '_max_entries': 300, '_cull_frequency': 3, 'key_prefix': '', 'version': 1, 'key_func': {}, '_dir': '/tmp/django_9b49xnhw/tmp14xshbwt'})
	obj_ins_pred = BaseCache(params={'default_timeout': pred_input['self']['default_timeout'], '_max_entries': pred_input['self']['_max_entries'], '_cull_frequency': pred_input['self']['_cull_frequency'], 'key_prefix': pred_input['self']['key_prefix'], 'version': pred_input['self']['version'], 'key_func': pred_input['self']['key_func'], '_dir': pred_input['self']['_dir']})
	assert obj_ins.get_many(keys = {}, version = 2)==obj_ins_pred.get_many(keys = pred_input['args']['key'], version = pred_input['kwargs']['version']), 'Prediction failed!'
 
