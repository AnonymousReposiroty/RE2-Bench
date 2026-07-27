# Problem: matplotlib@@matplotlib_ticker.py@@_is_decade_L2266
# Module: matplotlib.ticker
# Function: _is_decade
# Line: 2266

from matplotlib.ticker import _is_decade


def test_input(pred_input):
    assert _is_decade(x = 1e-07, base = 10, rtol = 1e-07)==_is_decade(x = pred_input['args']['x'], base = pred_input['args']['base'], rtol = pred_input['args']['rtol']), 'Prediction failed!'
    
    
    