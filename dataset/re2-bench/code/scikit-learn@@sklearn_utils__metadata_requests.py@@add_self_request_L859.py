from copy import deepcopy

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

    def add_self_request(self, obj):
        """Add `self` (as a :term:`consumer`) to the `MetadataRouter`.

        This method is used if the :term:`router` is also a :term:`consumer`, and hence
        the router itself needs to be included in the routing. The passed object
        can be an estimator or a
        :class:`~sklearn.utils.metadata_routing.MetadataRequest`.

        A router should add itself using this method instead of `add` since it
        should be treated differently than the other consumer objects to which metadata
        is routed by the router.

        Parameters
        ----------
        obj : object
            This is typically the router instance, i.e. `self` in a
            ``get_metadata_routing()`` implementation. It can also be a
            ``MetadataRequest`` instance.

        Returns
        -------
        self : MetadataRouter
            Returns `self`.
        """
        if getattr(obj, '_type', None) == 'metadata_request':
            self._self_request = deepcopy(obj)
        elif hasattr(obj, '_get_metadata_request'):
            self._self_request = deepcopy(obj._get_metadata_request())
        else:
            raise ValueError('Given `obj` is neither a `MetadataRequest` nor does it implement the required API. Inheriting from `BaseEstimator` implements the required API.')
        return self
