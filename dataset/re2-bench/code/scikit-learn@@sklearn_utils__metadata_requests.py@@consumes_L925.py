class MetadataRouter:
    """Coordinates metadata routing for a :term:`router` object.

    This class is used by :term:`meta-estimators` or functions that can route metadata,
    to handle their metadata routing. Routing information is stored in a
    dictionary-like structure of the form ``{"object_name":
    RouterMappingPair(mapping, router)}``, where ``mapping``
    is an instance of :class:`~sklearn.utils.metadata_routing.MethodMapping` and
    ``router`` is either a
    :class:`~sklearn.utils.metadata_routing.MetadataRequest` or another
    :class:`~sklearn.utils.metadata_routing.MetadataRouter` instance.

    .. versionadded:: 1.3

    Parameters
    ----------
    owner : object
        The object to which these requests belong.
    """
    _type = 'metadata_router'

    def __init__(self, owner):
        self._route_mappings = dict()
        self._self_request = None
        self.owner = owner

    def consumes(self, method, params):
        """Return params consumed as metadata in a :term:`router` or its sub-estimators.

        This method returns the subset of `params` that are consumed by the
        `method`. A `param` is considered consumed if it is used in the specified
        method of the :term:`router` itself or any of its sub-estimators (or their
        sub-estimators).

        .. versionadded:: 1.4

        Parameters
        ----------
        method : str
            The name of the method for which to determine consumed parameters.

        params : iterable of str
            An iterable of parameter names to test for consumption.

        Returns
        -------
        consumed_params : set of str
            A subset of parameters from `params` which are consumed by this method.
        """
        consumed_params = set()
        if self._self_request:
            consumed_params.update(self._self_request.consumes(method=method, params=params))
        for _, route_mapping in self._route_mappings.items():
            for caller, callee in route_mapping.mapping:
                if caller == method:
                    consumed_params.update(route_mapping.router.consumes(method=callee, params=params))
        return consumed_params
