class MethodMapping:
    """Stores the mapping between caller and callee methods for a :term:`router`.

    This class is primarily used in a ``get_metadata_routing()`` of a router
    object when defining the mapping between the router's methods and a sub-object (a
    sub-estimator or a scorer).

    Iterating through an instance of this class yields
    ``MethodPair(caller, callee)`` instances.

    .. versionadded:: 1.3
    """

    def __init__(self):
        self._routes = []

    def add(self, *, caller, callee):
        """Add a method mapping.

        Parameters
        ----------

        caller : str
            Parent estimator's method name in which the ``callee`` is called.

        callee : str
            Child object's method name. This method is called in ``caller``.

        Returns
        -------
        self : MethodMapping
            Returns self.
        """
        if caller not in METHODS:
            raise ValueError(f'Given caller:{caller} is not a valid method. Valid methods are: {METHODS}')
        if callee not in METHODS:
            raise ValueError(f'Given callee:{callee} is not a valid method. Valid methods are: {METHODS}')
        self._routes.append(MethodPair(caller=caller, callee=callee))
        return self
