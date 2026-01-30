# Problem: matplotlib@@matplotlib_pyplot.py@@switch_backend_L393
# Module: matplotlib.pyplot
# Function: switch_backend
# Line: 393

from matplotlib.pyplot import switch_backend


def test_input(pred_input):
    assert switch_backend(newbackend = 'Agg')==switch_backend(newbackend = pred_input['args']['newbackend']), 'Prediction failed!'
    