from xarray.coding.times import encode_cf_datetime

def test_input(pred_input):
	assert encode_cf_datetime(dates = '2000-01-11T00:00:00.000000000', units = 'days since 2000-01-01', calendar = 'gregorian')==encode_cf_datetime(dates = pred_input['args']['dates'], units = pred_input['args']['units'], calendar = pred_input['args']['calendar']), 'Prediction failed!'
