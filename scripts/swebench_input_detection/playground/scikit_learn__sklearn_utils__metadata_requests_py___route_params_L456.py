# Problem: scikit-learn@@sklearn.utils._metadata_requests.py@@_route_params_L456
# Module: sklearn.utils._metadata_requests
# Function: _route_params
# Line: 456

from sklearn.utils._metadata_requests import MethodMetadataRequest
from sklearn.utils._bunch import Bunch


def create_method_metadata_request():
    """
    Create a MethodMetadataRequest with the ground truth configuration:
    - owner: "ConsumingTransformer(registry=[])"
    - method: "fit_transform"
    - _requests: {"sample_weight": True, "metadata": True}
    """
    obj = MethodMetadataRequest(
        owner="ConsumingTransformer(registry=[])",
        method="fit_transform",
        requests={"sample_weight": True, "metadata": True}
    )
    return obj


def test_input(pred_input):
    # Create ground truth MethodMetadataRequest
    obj_gt = create_method_metadata_request()

    # Create predicted MethodMetadataRequest with pred_input values
    obj_pred = MethodMetadataRequest(
        owner=pred_input['self']['owner'],
        method=pred_input['self']['method'],
        requests=pred_input['self']['_requests']
    )

    # Ground truth args
    params_gt = {"sample_weight": [1], "metadata": "a"}
    parent_gt = "ColumnTransformer(transformers=[('trans', ConsumingTransformer(registry=[]),\n                                 [0])])"
    caller_gt = "fit_transform"

    # Predicted args
    params_pred = pred_input['args']['params']
    parent_pred = pred_input['args']['parent']
    caller_pred = pred_input['args']['caller']

    # Call _route_params on both and compare results
    result_gt = obj_gt._route_params(params=params_gt, parent=parent_gt, caller=caller_gt)
    result_pred = obj_pred._route_params(params=params_pred, parent=parent_pred, caller=caller_pred)

    # Compare Bunch results
    assert dict(result_gt) == dict(result_pred), 'Prediction failed!'
