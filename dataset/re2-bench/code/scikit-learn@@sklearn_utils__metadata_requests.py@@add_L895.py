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

    def add(self, *, method_mapping, **objs):
        """Add :term:`consumers <consumer>` to the `MetadataRouter`.

        The estimators that consume metadata are passed as named objects along with a
        method mapping, that defines how their methods relate to those of the
        :term:`router`.

        Parameters
        ----------
        method_mapping : MethodMapping
            The mapping between the child (:term:`consumer`) and the parent's
            (:term:`router`'s) methods.

        **objs : dict
            A dictionary of objects, whose requests are extracted by calling
            :func:`~sklearn.utils.metadata_routing.get_routing_for_object` on them.

        Returns
        -------
        self : MetadataRouter
            Returns `self`.
        """
        method_mapping = deepcopy(method_mapping)
        for name, obj in objs.items():
            self._route_mappings[name] = RouterMappingPair(mapping=method_mapping, router=get_routing_for_object(obj))
        return self
