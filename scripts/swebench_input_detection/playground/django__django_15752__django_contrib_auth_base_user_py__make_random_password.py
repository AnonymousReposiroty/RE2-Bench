# Problem: django__django-15752@@django.contrib.auth.base_user.py@@make_random_password
# Benchmark: Swebench
# Module: django.contrib.auth.base_user
# Function: make_random_password
from django.conf import settings
settings.configure(
    DEFAULT_CHARSET="utf-8",
)

import django
django.setup()

from django.contrib.auth.base_user import BaseUserManager


def test_input(pred_input):
    obj_ins = BaseUserManager()
    obj_ins._constructor_args = [[], {}]
    obj_ins.creation_counter = 476
    obj_ins.model = None
    obj_ins.name = None
    obj_ins._db = None
    obj_ins._hints = {}
    obj_ins_pred = BaseUserManager()
    obj_ins_pred._constructor_args = pred_input['self']['_constructor_args']
    obj_ins_pred.creation_counter = pred_input['self']['creation_counter']
    obj_ins_pred.model = pred_input['self']['model']
    obj_ins_pred.name = pred_input['self']['name']
    obj_ins_pred._db = pred_input['self']['_db']
    obj_ins_pred._hints = pred_input['self']['_hints']
    assert obj_ins.make_random_password()==obj_ins_pred.make_random_password(), 'Prediction failed!'


obj_ins = BaseUserManager()
obj_ins._constructor_args = [[], {}]
obj_ins.creation_counter = 476
obj_ins.model = None
obj_ins.name = None
obj_ins._db = None
obj_ins._hints = {}
print(obj_ins.make_random_password())