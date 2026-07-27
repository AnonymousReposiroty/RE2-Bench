import django
from django.conf import settings

settings.configure(
    DEBUG=False,
    USE_I18N=True,
    USE_L10N=True,
    USE_TZ=True,
)
django.setup()
from django.utils.formats import get_format

def test_input(pred_input):
	assert get_format(format_type = 'jS \\o\\f F', use_l10n = None)==get_format(format_type = pred_input['args']['format_type'], use_l10n = pred_input['kwargs']['use_l10n']), 'Prediction failed!'
 
