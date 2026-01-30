from django.forms.widgets import SelectDateWidget
import django
from django.conf import settings

settings.configure(
    DEBUG=False,
    USE_I18N=True,
    USE_L10N=True,
    USE_TZ=True,
)
django.setup()
def test_input(pred_input):
	obj_ins = SelectDateWidget(attrs = {}, years = [2007], months = {'1': {'_proxy____args': ['January'], '_proxy____kw': {}, '_proxy____prepared': True}, '2': {'_proxy____args': ['February'], '_proxy____kw': {}, '_proxy____prepared': True}, '3': {'_proxy____args': ['March'], '_proxy____kw': {}, '_proxy____prepared': True}, '4': {'_proxy____args': ['April'], '_proxy____kw': {}, '_proxy____prepared': True}, '5': {'_proxy____args': ['May'], '_proxy____kw': {}, '_proxy____prepared': True}, '6': {'_proxy____args': ['June'], '_proxy____kw': {}, '_proxy____prepared': True}, '7': {'_proxy____args': ['July'], '_proxy____kw': {}, '_proxy____prepared': True}, '8': {'_proxy____args': ['August'], '_proxy____kw': {}, '_proxy____prepared': True}, '9': {'_proxy____args': ['September'], '_proxy____kw': {}, '_proxy____prepared': True}, '10': {'_proxy____args': ['October'], '_proxy____kw': {}, '_proxy____prepared': True}, '11': {'_proxy____args': ['November'], '_proxy____kw': {}, '_proxy____prepared': True}, '12': {'_proxy____args': ['December'], '_proxy____kw': {}, '_proxy____prepared': True}})
	obj_ins_pred = SelectDateWidget(attrs = pred_input['self']['attrs'], years = pred_input['self']['years'], months = pred_input['self']['months'])
	assert obj_ins.get_context(name = 'mydate', value = '', attrs = None)==obj_ins_pred.get_context(name = pred_input['args']['name'], value = pred_input['args']['value'], attrs = pred_input['args']['attrs']), 'Prediction failed!'