# Problem: django__django-11133@@django.http.response.py@@make_bytes
# Benchmark: Swebench
# Module: django.http.response
# Function: make_bytes
from django.conf import settings
settings.configure(
    DEFAULT_CHARSET="utf-8",
)

import django
django.setup()

from django.http.response import HttpResponseBase


def test_input(pred_input):
    obj_ins = HttpResponseBase()
    obj_ins._headers = {'content-type': ['Content-Type', 'text/html; charset=utf-8']}
    obj_ins._closable_objects = None
    obj_ins._handler_class = None
    obj_ins.cookies = {}
    obj_ins.closed = False
    obj_ins._reason_phrase = None
    obj_ins._charset = None
    obj_ins._container = None
    obj_ins_pred = HttpResponseBase()
    obj_ins_pred._headers = pred_input['self']['_headers']
    obj_ins_pred._closable_objects = pred_input['self']['_closable_objects']
    obj_ins_pred._handler_class = pred_input['self']['_handler_class']
    obj_ins_pred.cookies = pred_input['self']['cookies']
    obj_ins_pred.closed = pred_input['self']['closed']
    obj_ins_pred._reason_phrase = pred_input['self']['_reason_phrase']
    obj_ins_pred._charset = pred_input['self']['_charset']
    obj_ins_pred._container = pred_input['self']['_container']
    assert obj_ins.make_bytes(value = 'baz\n')==obj_ins_pred.make_bytes(value = pred_input['args']['value']), 'Prediction failed!'
    
