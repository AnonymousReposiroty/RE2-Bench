from sklearn.utils.metadata_routing import (
    MetadataRequest,
    MetadataRouter,
    MethodMapping,
    _MetadataRequester,
    _raise_for_params,
    _routing_enabled,
    get_routing_for_object,
    process_routing,
)

class _PassthroughScorer(_MetadataRequester):

    def __init__(self, estimator):
        self._estimator = estimator

    def get_metadata_routing(self):
        """Get requested data properties.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Returns
        -------
        routing : MetadataRouter
            A :class:`~utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        return get_routing_for_object(self._estimator)
