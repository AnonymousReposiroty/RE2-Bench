# Problem: pydata__xarray-6135@@xarray.coding.cftime_offsets.py@@__mul__
# Benchmark: Swebench
# Module: xarray.coding.cftime_offsets
# Function: __mul__

from xarray.coding.cftime_offsets import QuarterOffset


def test_input(pred_input):
    QuarterOffset._default_month = 2
    obj_ins = QuarterOffset()
    obj_ins.n = 1
    obj_ins.month = 2
    
    obj_ins_pred = QuarterOffset()
    obj_ins_pred.n = pred_input['self']['n']
    obj_ins_pred.month = pred_input['self']['month']
    assert obj_ins.__mul__(other = -1)==obj_ins_pred.__mul__(other = pred_input['args']['other']), 'Prediction failed!'