class MetadataRequest:
    """Contains the metadata request info of a consumer.

    Instances of `MethodMetadataRequest` are used in this class for each
    available method under `metadatarequest.{method}`.

    Consumer-only classes such as simple estimators return a serialized
    version of this class as the output of `get_metadata_routing()`.

    .. versionadded:: 1.3

    Parameters
    ----------
    owner : object
        The object to which these requests belong.
    """
    _type = 'metadata_request'

    def __init__(self, owner):
        self.owner = owner
        for method in SIMPLE_METHODS:
            setattr(self, method, MethodMetadataRequest(owner=owner, method=method))

    def _route_params(self, *, params, method, parent, caller):
        """Prepare the given parameters to be passed to the method.

        The output of this method can be used directly as the input to the
        corresponding method as extra keyword arguments to pass metadata.

        Parameters
        ----------
        params : dict
            A dictionary of provided metadata.

        method : str
            The name of the method for which the parameters are requested and
            routed.

        parent : object
            Parent class object, that routes the metadata.

        caller : str
            Method from the parent class object, where the metadata is routed from.

        Returns
        -------
        params : Bunch
            A :class:`~sklearn.utils.Bunch` of {metadata: value} which can be given to
            the corresponding method.
        """
        return getattr(self, method)._route_params(params=params, parent=parent, caller=caller)
