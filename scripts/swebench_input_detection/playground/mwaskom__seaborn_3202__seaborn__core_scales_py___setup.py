from seaborn._core.scales import ContinuousBase
import pandas as pd


def test_input(pred_input):
	obj_ins = ContinuousBase(values = None, norm = None)
	obj_ins_pred = ContinuousBase(values = pred_input['self']['values'], norm = pred_input['self']['norm'])
	d = pd.Series(name="x")
	c = pd.Series(name=pred_input['args']['data']["_name"])
	assert obj_ins._setup(data = d, prop = {'variable': 'y'})==obj_ins_pred._setup(data = c, prop = pred_input['args']['prop']), 'Prediction failed!'