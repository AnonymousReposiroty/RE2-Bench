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

    def _get_param_names(self, *, method, return_alias, ignore_self_request):
        """Get names of all metadata that can be consumed or routed by specified             method.

        This method returns the names of all metadata, even the ``False``
        ones.

        Parameters
        ----------
        method : str
            The name of the method for which metadata names are requested.

        return_alias : bool
            Controls whether original or aliased names should be returned,
            which only applies to the stored `self`. If no `self` routing
            object is stored, this parameter has no effect.

        ignore_self_request : bool
            If `self._self_request` should be ignored. This is used in `_route_params`.
            If ``True``, ``return_alias`` has no effect.

        Returns
        -------
        names : set of str
            A set of strings with the names of all metadata.
        """
        res = set()
        if self._self_request and (not ignore_self_request):
            res = res.union(self._self_request._get_param_names(method=method, return_alias=return_alias))
        for name, route_mapping in self._route_mappings.items():
            for caller, callee in route_mapping.mapping:
                if caller == method:
                    res = res.union(route_mapping.router._get_param_names(method=callee, return_alias=True, ignore_self_request=False))
        return res
