from sklearn.utils._bunch import Bunch

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

    def route_params(self, *, caller, params):
        """Get the values of metadata requested by :term:`consumers <consumer>`.

        Returns a :class:`~sklearn.utils.Bunch` containing the metadata that this
        :term:`router`'s `caller` method needs to route, organized by each
        :term:`consumer` and their corresponding methods.

        This can be used to pass the required metadata to corresponding methods in
        consumers.

        Parameters
        ----------
        caller : str
            The name of the :term:`router`'s method through which the metadata is
            routed. For example, if called inside the :term:`fit` method of a router,
            this would be `"fit"`.

        params : dict
            A dictionary of provided metadata.

        Returns
        -------
        params : Bunch
            A :class:`~sklearn.utils.Bunch` of the form
            ``{"object_name": {"method_name": {metadata: value}}}``.
        """
        if self._self_request:
            self._self_request._check_warnings(params=params, method=caller)
        res = Bunch()
        for name, route_mapping in self._route_mappings.items():
            router, mapping = (route_mapping.router, route_mapping.mapping)
            res[name] = Bunch()
            for _caller, _callee in mapping:
                if _caller == caller:
                    res[name][_callee] = router._route_params(params=params, method=_callee, parent=self.owner, caller=caller)
        return res
