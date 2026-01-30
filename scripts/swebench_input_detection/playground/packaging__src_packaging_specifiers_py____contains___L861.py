# Problem: packaging@@src_packaging_specifiers.py@@__contains___L861
# Module: packaging.specifiers
# Function: __contains__
# Line: 861

from packaging.specifiers import SpecifierSet


def test_input(pred_input):
    obj_ins = SpecifierSet()
    obj_ins._specs = 'frozenset()'
    obj_ins._prereleases = True
    obj_ins_pred = SpecifierSet()
    obj_ins_pred._specs = pred_input['self']['_specs']
    obj_ins_pred._prereleases = pred_input['self']['_prereleases']
    assert obj_ins.__contains__(item = "<Version('1.0a1')>")==obj_ins_pred.__contains__(item = pred_input['args']['item']), 'Prediction failed!'