from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    get_routing_for_object,
    process_routing,
)
from sklearn.utils.validation import (
    _check_method_params,
    _check_sample_weight,
    _estimator_has,
    check_is_fitted,
    has_fit_parameter,
    validate_data,
)

def _consumes_sample_weight(estimator):
    if _routing_enabled():
        request_or_router = get_routing_for_object(estimator)
        consumes_sample_weight = request_or_router.consumes("fit", ("sample_weight",))
    else:
        consumes_sample_weight = has_fit_parameter(estimator, "sample_weight")
    return consumes_sample_weight
