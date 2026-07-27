class MethodMetadataRequest:
    """Container for metadata requests associated with a single method.

    Instances of this class get used within a :class:`MetadataRequest` - one per each
    public method (`fit`, `transform`, ...) that its owning consumer has.

    .. versionadded:: 1.3

    Parameters
    ----------
    owner : object
        The object owning these requests.

    method : str
        The name of the method to which these requests belong.

    requests : dict of {str: bool, None or str}, default=None
        The initial requests for this method.
    """

    def __init__(self, owner, method, requests=None):
        self._requests = requests or dict()
        self.owner = owner
        self.method = method

    def _get_param_names(self, return_alias):
        """Get names of all metadata that can be consumed or routed by this method.

        This method returns the names of all metadata, even the ``False``
        ones.

        Parameters
        ----------
        return_alias : bool
            Controls whether original or aliased names should be returned. If
            ``False``, aliases are ignored and original names are returned.

        Returns
        -------
        names : set of str
            A set of strings with the names of all metadata.
        """
        return set((alias if return_alias and (not request_is_valid(alias)) else prop for prop, alias in self._requests.items() if not request_is_valid(alias) or alias is not False))
