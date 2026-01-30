# Problem: django__django-16560@@django.db.models.constraints.py@@__eq__
# Benchmark: Swebench
# Module: django.db.models.constraints
# Function: __eq__
from django.conf import settings
settings.configure(
    DEFAULT_CHARSET="utf-8",
)

import django
django.setup()

from django.db.models.constraints import CheckConstraint



def test_input(pred_input):
    
    assert pred_input['self']['check'] == pred_input['args']['other']['check'], 'Prediction failed!'
    assert pred_input['self']['name'] == pred_input['args']['other']['name'], 'Prediction failed!'
    assert pred_input['self']['violation_error_code'] == pred_input['args']['other']['violation_error_code'], 'Prediction failed!'
    assert pred_input['self']['violation_error_message'] == pred_input['args']['other']['violation_error_message'], 'Prediction failed!'

    