import numpy as np

# Problem: scikit-learn@@sklearn_utils__metadata_requests.py@@_get_metadata_request_L1523
# Module: sklearn.utils._metadata.requests
# Function: _get_metadata_request
# Line: 1523

from sklearn.utils._metadata.requests import _MetadataRequester


def test_input(pred_input):
    assert {'fit': {'sample_weight': True, 'metadata': True}, 'transform': {'sample_weight': True, 'metadata': True}, 'inverse_transform': {'sample_weight': None, 'metadata': None}}==pred_input['self']['_metadata_request'], 'Prediction failed!'