# Problem: django__django-11170@@django.views.debug.py@@default_urlconf
# Benchmark: Swebench
# Module: django.views.debug
# Function: default_urlconf
import django
from django.conf import settings

settings.configure(
    DEBUG=False,
    USE_I18N=True,
    USE_L10N=True,
    USE_TZ=True,
)
django.setup()
from django.views.debug import default_urlconf


def test_input(pred_input):
    assert default_urlconf(request = None)==default_urlconf(request = pred_input['args']['request']), 'Prediction failed!'
    