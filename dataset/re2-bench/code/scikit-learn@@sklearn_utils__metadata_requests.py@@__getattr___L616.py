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

    def __getattr__(self, name):
        if name not in COMPOSITE_METHODS:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        requests = {}
        for method in COMPOSITE_METHODS[name]:
            mmr = getattr(self, method)
            existing = set(requests.keys())
            upcoming = set(mmr.requests.keys())
            common = existing & upcoming
            conflicts = [key for key in common if requests[key] != mmr._requests[key]]
            if conflicts:
                raise ValueError(f"Conflicting metadata requests for {', '.join(conflicts)} while composing the requests for {name}. Metadata with the same name for methods {', '.join(COMPOSITE_METHODS[name])} should have the same request value.")
            requests.update(mmr._requests)
        return MethodMetadataRequest(owner=self.owner, method=name, requests=requests)
