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

    def _get_param_names(self, method, return_alias, ignore_self_request=None):
        """Get names of all metadata that can be consumed or routed by specified             method.

        This method returns the names of all metadata, even the ``False``
        ones.

        Parameters
        ----------
        method : str
            The name of the method for which metadata names are requested.

        return_alias : bool
            Controls whether original or aliased names should be returned. If
            ``False``, aliases are ignored and original names are returned.

        ignore_self_request : bool
            Ignored. Present for API compatibility.

        Returns
        -------
        names : set of str
            A set of strings with the names of all metadata.
        """
        return getattr(self, method)._get_param_names(return_alias=return_alias)
