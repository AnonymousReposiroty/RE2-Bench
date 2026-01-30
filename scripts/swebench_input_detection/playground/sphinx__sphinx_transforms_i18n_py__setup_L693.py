# Problem: sphinx@@sphinx_transforms_i18n.py@@setup_L693
# Module: sphinx.transforms.i18n
# Function: setup
# Line: 693


def test_input(pred_input):
    assert "<SphinxTestApp buildername='html'>" == pred_input['args']['app'], 'Prediction failed!'