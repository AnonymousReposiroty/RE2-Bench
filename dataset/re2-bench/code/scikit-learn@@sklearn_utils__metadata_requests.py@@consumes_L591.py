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

    def consumes(self, method, params):
        """Return params consumed as metadata in a :term:`consumer`.

        This method returns the subset of given `params` that are consumed by the
        given `method`. It can be used to check if parameters are used as metadata in
        the specified method of the :term:`consumer` that owns this `MetadataRequest`
        instance.

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
            A subset of parameters from `params` which are consumed by the given method.
        """
        return getattr(self, method)._consumes(params=params)
