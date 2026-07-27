# Problem: sphinx@@sphinx_environment_collectors_metadata.py@@setup_L71
# Module: sphinx.environment.collectors.metadata
# Function: setup
# Line: 71
def test_input(pred_input):
    assert "<SphinxTestApp buildername='html'>"== pred_input['args']['app'], 'Prediction failed!'