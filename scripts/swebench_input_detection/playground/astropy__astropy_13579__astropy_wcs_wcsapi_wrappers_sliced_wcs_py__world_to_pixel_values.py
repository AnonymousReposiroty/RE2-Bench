from astropy.wcs.wcsapi.wrappers.sliced_wcs import SlicedLowLevelWCS

def test_input(pred_input):
	obj_ins = SlicedLowLevelWCS(_wcs = {'_init_kwargs': {'keysel': None, 'colsel': None}, 'naxis': 3, '_naxis': None, '_pixel_bounds': None}, _slices_array = None, _slices_pixel = None, _pixel_keep = '[0 1]', _world_keep = '[0 1]')
	obj_ins_pred = SlicedLowLevelWCS(_wcs = pred_input['self']['_wcs'], _slices_array = pred_input['self']['_slices_array'], _slices_pixel = pred_input['self']['_slices_pixel'], _pixel_keep = pred_input['self']['_pixel_keep'], _world_keep = pred_input['self']['_world_keep'])
	assert obj_ins.world_to_pixel_values()==obj_ins_pred.world_to_pixel_values(), 'Prediction failed!'
 
