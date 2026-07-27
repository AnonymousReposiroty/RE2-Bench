# Problem: sphinx-doc__sphinx-8731@@sphinx.ext.napoleon.docstring.py@@_convert_type_spec
# Benchmark: Swebench
# Module: sphinx.ext.napoleon.docstring
# Function: _convert_type_spec

from sphinx.ext.napoleon.docstring import _convert_type_spec


def test_input(pred_input):
    assert _convert_type_spec(_type = 'str', translations = {})==_convert_type_spec(_type = pred_input['args']['_type'], translations = pred_input['args']['translations']), 'Prediction failed!'
    