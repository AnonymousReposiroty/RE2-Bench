# Problem: django__django-12276@@django.forms.widgets.py@@use_required_attribute
# Benchmark: Swebench
# Module: django.forms.widgets
# Function: use_required_attribute

from django.forms.widgets import FileInput


def test_input(pred_input):
    obj_ins = FileInput()
    obj_ins.attrs = {}
    obj_ins_pred = FileInput()
    obj_ins_pred.attrs = pred_input['self']['attrs']
    assert obj_ins.use_required_attribute(initial = None)==obj_ins_pred.use_required_attribute(initial = pred_input['args']['initial']), 'Prediction failed!'