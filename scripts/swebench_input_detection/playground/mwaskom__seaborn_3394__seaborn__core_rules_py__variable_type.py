from seaborn._core.rules import variable_type
import pandas as pd

def test_input(pred_input):
	assert variable_type(vector = pd.Series(name=None), boolean_type = 'boolean')==variable_type(vector =pd.Series(name=pred_input['args']['vector']['_name']), boolean_type = pred_input['kwargs']['boolean_type']), 'Prediction failed!'
 
 
