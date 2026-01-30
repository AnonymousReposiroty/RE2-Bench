# Problem: attrs@@src_attr__make.py@@add_repr_L992
# Module: attr._make
# Function: add_repr
# Line: 992

from attr._make import _ClassBuilder


def test_input(pred_input):
    obj_ins = <_ClassBuilder(cls=Concrete)>
    obj_ins_pred = pred_input['self']
    assert obj_ins.add_repr(ns = None)==obj_ins_pred.add_repr(ns = pred_input['args']['ns']), 'Prediction failed!'