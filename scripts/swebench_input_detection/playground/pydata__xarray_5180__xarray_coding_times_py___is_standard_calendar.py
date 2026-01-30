# Problem: pydata__xarray-5180@@xarray.coding.times.py@@_is_standard_calendar
# Benchmark: Swebench
# Module: xarray.coding.times
# Function: _is_standard_calendar

from xarray.coding.times import _is_standard_calendar


def test_input(pred_input):
    assert _is_standard_calendar(calendar = 'GREGORIAN')==_is_standard_calendar(calendar = pred_input['args']['calendar']), 'Prediction failed!'
