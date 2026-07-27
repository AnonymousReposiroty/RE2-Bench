import inspect
from collections import defaultdict, namedtuple
from typing import TYPE_CHECKING, Optional, Union

class _MetadataRequester:
    """Mixin class for adding metadata request functionality.

    ``BaseEstimator`` inherits from this Mixin.

    .. versionadded:: 1.3
    """
    if TYPE_CHECKING:

        def set_fit_request(self, **kwargs):
            pass

        def set_partial_fit_request(self, **kwargs):
            pass

        def set_predict_request(self, **kwargs):
            pass

        def set_predict_proba_request(self, **kwargs):
            pass

        def set_predict_log_proba_request(self, **kwargs):
            pass

        def set_decision_function_request(self, **kwargs):
            pass

        def set_score_request(self, **kwargs):
            pass

        def set_split_request(self, **kwargs):
            pass

        def set_transform_request(self, **kwargs):
            pass

        def set_inverse_transform_request(self, **kwargs):
            pass

    @classmethod
    def _get_class_level_metadata_request_values(cls, method: str):
        """Get class level metadata request values.

        This method first checks the `method`'s signature for passable metadata and then
        updates these with the metadata request values set at class level via the
        ``__metadata_request__{method}`` class attributes.

        This method (being a class-method), does not take request values set at
        instance level into account.
        """
        if not hasattr(cls, method) or not inspect.isfunction(getattr(cls, method)):
            return dict()
        signature_items = list(inspect.signature(getattr(cls, method)).parameters.items())[1:]
        params = defaultdict(str, {param_name: None for param_name, param_info in signature_items if param_name not in {'X', 'y', 'Y', 'Xt', 'yt'} and param_info.kind not in {param_info.VAR_POSITIONAL, param_info.VAR_KEYWORD}})
        substr = f'__metadata_request__{method}'
        for base_class in reversed(inspect.getmro(cls)):
            base_class_items = vars(base_class).copy().items()
            for attr, value in base_class_items:
                if substr not in attr:
                    continue
                for prop, alias in value.items():
                    if prop not in params and alias == UNUSED:
                        raise ValueError(f"Trying to remove parameter {prop} with UNUSED which doesn't exist.")
                    params[prop] = alias
        return {param: alias for param, alias in params.items() if alias is not UNUSED}

    def _get_metadata_request(self):
        """Get requested metadata for the instance.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        request : MetadataRequest
            A :class:`~sklearn.utils.metadata_routing.MetadataRequest` instance.
        """
        if hasattr(self, '_metadata_request'):
            requests = get_routing_for_object(self._metadata_request)
        else:
            requests = MetadataRequest(owner=self)
            for method in SIMPLE_METHODS:
                setattr(requests, method, MethodMetadataRequest(owner=self, method=method, requests=self._get_class_level_metadata_request_values(method)))
        return requests

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRequest
            A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating
            routing information.
        """
        return self._get_metadata_request()
