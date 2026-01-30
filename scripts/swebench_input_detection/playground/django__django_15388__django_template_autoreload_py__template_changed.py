# Problem: django__django-15388@@django.template.autoreload.py@@template_changed
# Benchmark: Swebench
# Module: django.template.autoreload
# Function: template_changed
from pathlib import Path
from django.conf import settings
settings.configure(
    DEFAULT_CHARSET="utf-8",
)

import django
django.setup()
from django.template.autoreload import template_changed


def test_input(pred_input):
    assert template_changed(sender = None, file_path = Path('/testbed/tests/template_tests/templates/index.html'))==template_changed(sender = pred_input['args']['sender'], file_path = pred_input['args']['file_path']), 'Prediction failed!'
