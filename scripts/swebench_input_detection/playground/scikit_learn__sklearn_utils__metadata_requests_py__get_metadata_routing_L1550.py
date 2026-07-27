# Problem: scikit-learn@@sklearn.utils._metadata_requests.py@@get_metadata_routing_L1550
# Module: sklearn.utils._metadata_requests
# Function: get_metadata_routing
# Line: 1550

from sklearn.utils._metadata_requests import (
    _MetadataRequester,
    MetadataRequest,
    MethodMetadataRequest,
    SIMPLE_METHODS,
)


class ConsumingTransformer(_MetadataRequester):
    """Mock class that mimics ConsumingTransformer for testing."""

    def __init__(self, registry=None):
        self.registry = registry if registry is not None else []

    def fit(self, X, y=None, sample_weight=None, metadata=None):
        pass

    def transform(self, X, sample_weight=None, metadata=None):
        pass

    def inverse_transform(self, X, sample_weight=None, metadata=None):
        pass


def create_metadata_request(owner):
    """
    Create a MetadataRequest with the ground truth configuration:
    - fit: {'sample_weight': True, 'metadata': True}
    - transform: {'sample_weight': True, 'metadata': True}
    - inverse_transform: {'sample_weight': None, 'metadata': None}
    """
    request = MetadataRequest(owner=owner)
    # Set fit requests
    request.fit._requests = {'sample_weight': True, 'metadata': True}
    # Set transform requests
    request.transform._requests = {'sample_weight': True, 'metadata': True}
    # Set inverse_transform requests
    request.inverse_transform._requests = {'sample_weight': None, 'metadata': None}
    return request


def create_consuming_transformer():
    """
    Create a ConsumingTransformer with the ground truth configuration.
    """
    obj = ConsumingTransformer(
        registry=["ConsumingTransformer(('registry', [ConsumingTransformer(('registry', [...]))]))"]
    )
    obj._metadata_request = create_metadata_request(obj)
    return obj


def test_input(pred_input):
    # Create ground truth object
    obj_gt = create_consuming_transformer()

    # Create predicted object with pred_input values
    obj_pred = ConsumingTransformer(
        registry=pred_input['self']['registry']
    )

    # Parse _metadata_request from string representation
    metadata_request_str = pred_input['self']['_metadata_request']
    metadata_request_dict = eval(metadata_request_str)

    # Create MetadataRequest and set the requests
    obj_pred._metadata_request = MetadataRequest(owner=obj_pred)
    for method, requests in metadata_request_dict.items():
        if hasattr(obj_pred._metadata_request, method):
            getattr(obj_pred._metadata_request, method)._requests = requests

    # Call get_metadata_routing on both
    result_gt = obj_gt.get_metadata_routing()
    result_pred = obj_pred.get_metadata_routing()

    # Compare results by checking the requests for each method
    for method in SIMPLE_METHODS:
        gt_requests = getattr(result_gt, method)._requests
        pred_requests = getattr(result_pred, method)._requests
        assert gt_requests == pred_requests, f'Prediction failed for method {method}!'
