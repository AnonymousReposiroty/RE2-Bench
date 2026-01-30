# Problem: django__django-15731@@django.db.models.manager.py@@create_method
# Benchmark: Swebench
# Module: django.db.models.manager
# Function: create_method

from django.db.models.manager import BaseManager


def test_input(pred_input):
    obj_ins = BaseManager()
    obj_ins_pred = BaseManager()
    assert ('values_list', {}) == (pred_input['args']['name'], pred_input['args']['method']), 'Prediction failed!'
    