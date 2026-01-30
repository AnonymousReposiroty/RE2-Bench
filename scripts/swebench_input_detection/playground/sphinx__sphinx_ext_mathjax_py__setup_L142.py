# Problem: sphinx@@sphinx_ext_mathjax.py@@setup_L142
# Module: sphinx.ext.mathjax
# Function: setup
# Line: 142


def test_input(pred_input):
    assert "<SphinxTestApp buildername='html'>"== pred_input['args']['app'], 'Prediction failed!'