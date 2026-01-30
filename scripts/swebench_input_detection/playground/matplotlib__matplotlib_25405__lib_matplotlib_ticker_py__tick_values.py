from matplotlib.ticker import LogLocator

def test_input(pred_input):
	obj_ins = LogLocator(numdecs = 4, numticks = 3)
	obj_ins._base=10.0
	obj_ins._subs='[1.]'
	obj_ins_pred = LogLocator(numdecs = pred_input['self']['numdecs'], numticks = pred_input['self']['numticks'])
	obj_ins_pred._base=pred_input['self']['_base']
	obj_ins_pred._subs=pred_input['self']['_subs']
	assert obj_ins.tick_values(vmin = 6, vmax = 150)==obj_ins_pred.tick_values(vmin = pred_input['args']['vmin'], vmax = pred_input['args']['vmax']), 'Prediction failed!'