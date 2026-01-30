from xarray.coding.times import decode_cf_datetime
import numpy as np

def test_input(pred_input):
	assert decode_cf_datetime(num_dates = np.array([12300, 12301, 12302, 12303, 12304]), units = 'hour since 1680-01-01 00:00:00.500000', calendar = 'standard')==decode_cf_datetime(num_dates = pred_input['args']['num_dates'], units = pred_input['args']['units'], calendar = pred_input['args']['calendar']), 'Prediction failed!'
 
 
a = decode_cf_datetime(num_dates = np.array([12300, 12301, 12302, 12303, 12304]), units = 'hour since 1680-01-01 00:00:00.500000', calendar = 'standard')
print(a)