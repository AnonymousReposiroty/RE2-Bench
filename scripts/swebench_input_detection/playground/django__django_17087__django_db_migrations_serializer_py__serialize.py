# Problem: django__django-17087@@django.db.migrations.serializer.py@@serialize
# Benchmark: Swebench
# Module: django.db.migrations.serializer
# Function: serialize

from django.db.migrations.serializer import FunctionTypeSerializer
from time import time

def test_input(pred_input):
    obj_ins = FunctionTypeSerializer(value=time)
    obj_ins.value = time
    
    if "time" in pred_input['self']['value']:
        obj_ins_pred = FunctionTypeSerializer(value=time)
    else:
        obj_ins_pred = FunctionTypeSerializer(value=None)
    assert obj_ins.serialize()==obj_ins_pred.serialize(), 'Prediction failed!'
