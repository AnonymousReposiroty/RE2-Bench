# Problem: sphinx-doc__sphinx-9658@@sphinx.ext.autodoc.mock.py@@_make_subclass
# Benchmark: Swebench
# Module: sphinx.ext.autodoc.mock
# Function: _make_subclass

from sphinx.ext.autodoc.mock import _make_subclass


def test_input(pred_input):
    assert _make_subclass(name = 'secret', module = 'unknown')==_make_subclass(name = pred_input['args']['name'], module = pred_input['args']['module']), 'Prediction failed!'
    
_make_subclass(name = 'secret', module = 'unknown')