class ResourceMeta:
    """
    An object containing metadata about a resource.
    """

    def __init__(
        self,
        service_name,
        identifiers=None,
        client=None,
        data=None,
        resource_model=None,
    ):
        #: (``string``) The service name, e.g. 's3'
        self.service_name = service_name

        if identifiers is None:
            identifiers = []
        #: (``list``) List of identifier names
        self.identifiers = identifiers

        #: (:py:class:`~botocore.client.BaseClient`) Low-level Botocore client
        self.client = client
        #: (``dict``) Loaded resource data attributes
        self.data = data

        # The resource model for that resource
        self.resource_model = resource_model

    def __repr__(self):
        return f'ResourceMeta(\'{self.service_name}\', identifiers={self.identifiers})'

    def __eq__(self, other):
        # Two metas are equal if their components are all equal
        if other.__class__.__name__ != self.__class__.__name__:
            return False

        return self.__dict__ == other.__dict__

    def copy(self):
        """
        Create a copy of this metadata object.
        """
        params = self.__dict__.copy()
        service_name = params.pop('service_name')
        return ResourceMeta(service_name, **params)
