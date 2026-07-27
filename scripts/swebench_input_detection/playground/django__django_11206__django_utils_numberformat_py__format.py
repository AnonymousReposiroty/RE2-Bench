import django
from django.conf import settings

settings.configure(
    DEBUG=False,
    USE_I18N=True,
    USE_L10N=True,
    USE_TZ=True,
)
django.setup()

from django.utils.numberformat import format

def test_input(pred_input):
	assert format(number = -359538626972463141629054847463408713596141135051689993197834953606314521560057077521179117265533756343080917907028764928468642653778928365536935093407075033972099821153102564152490980180778657888151737016910267884609166473806445896331617118664246696549595652408289446337476354361838599762500808052368249716736, decimal_sep = '.')==format(number = pred_input['args']['number'], decimal_sep = pred_input['args']['decimal_sep']), 'Prediction failed!'