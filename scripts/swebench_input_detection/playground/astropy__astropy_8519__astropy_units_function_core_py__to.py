from astropy.units.function.core import FunctionUnitBase

def test_input(pred_input):
	obj_ins = FunctionUnitBase(_physical_unit = {'_names': None, '_short_names': None, '_long_names': None, '_format': {}, '__doc__': 'meter: base unit of length in SI', '_hash': 2181990727224913881}, _function_unit = {'_bases': None, '_powers': None, '_scale': -2.0, '_decomposed_cache': {'_scale': 0.8, '_bases': None, '_powers': None}})
	obj_ins_pred = FunctionUnitBase(_physical_unit = pred_input['self']['_physical_unit'], _function_unit = pred_input['self']['_function_unit'])
	assert obj_ins.to(other = {'_names': None, '_short_names': None, '_long_names': None, '_format': {}, '__doc__': 'meter: base unit of length in SI', '_hash': 2181990727224913881}, value = '2.5', equivalencies = None)==obj_ins_pred.to(other = pred_input['args']['other'], value = pred_input['args']['value'], equivalencies = pred_input['kwargs']['equivalencies']), 'Prediction failed!'
