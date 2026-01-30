# Problem: django__django-16343@@django.core.signing.py@@loads
# Benchmark: Swebench
# Module: django.core.signing
# Function: loads

from django.core.signing import loads
from django.conf import settings
import django

settings.configure(
    DEBUG=False,
    USE_I18N=True,
    USE_L10N=True,
    USE_TZ=True,
    SECRET_KEY="dummy"
)
django.setup()

def test_input(pred_input):
    assert 'eyJhIjoiZGljdGlvbmFyeSJ9:1u7SIo:wF3XgnVXy3Uny-ff_G2pL1VcbThKPy56l3Vn9gnoLso' ==  pred_input['args']['s'], 'Prediction failed!'
    
    
