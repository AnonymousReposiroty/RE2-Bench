from botocore import xform_name

class ResourceModel:
    """
    A model representing a resource, defined via a JSON description
    format. A resource has identifiers, attributes, actions,
    sub-resources, references and collections. For more information
    on resources, see :ref:`guide_resources`.

    :type name: string
    :param name: The name of this resource, e.g. ``sqs`` or ``Queue``
    :type definition: dict
    :param definition: The JSON definition
    :type resource_defs: dict
    :param resource_defs: All resources defined in the service
    """

    def __init__(self, name, definition, resource_defs):
        self._definition = definition
        self._resource_defs = resource_defs
        self._renamed = {}
        self.name = name
        self.shape = definition.get('shape')

    def _get_name(self, category, name, snake_case=True):
        """
        Get a possibly renamed value given a category and name. This
        uses the rename map set up in ``load_rename_map``, so that
        method must be called once first.

        :type category: string
        :param category: The value type, such as 'identifier' or 'action'
        :type name: string
        :param name: The original name of the value
        :type snake_case: bool
        :param snake_case: True (default) if the name should be snake cased.
        :rtype: string
        :return: Either the renamed value if it is set, otherwise the
                 original name.
        """
        if snake_case:
            name = xform_name(name)
        return self._renamed.get((category, name), name)

    def get_attributes(self, shape):
        """
        Get a dictionary of attribute names to original name and shape
        models that represent the attributes of this resource. Looks
        like the following:

            {
                'some_name': ('SomeName', <Shape...>)
            }

        :type shape: botocore.model.Shape
        :param shape: The underlying shape for this resource.
        :rtype: dict
        :return: Mapping of resource attributes.
        """
        attributes = {}
        identifier_names = [i.name for i in self.identifiers]
        for name, member in shape.members.items():
            snake_cased = xform_name(name)
            if snake_cased in identifier_names:
                continue
            snake_cased = self._get_name('attribute', snake_cased, snake_case=False)
            attributes[snake_cased] = (name, member)
        return attributes

    @property
    def identifiers(self):
        """
        Get a list of resource identifiers.

        :type: list(:py:class:`Identifier`)
        """
        identifiers = []
        for item in self._definition.get('identifiers', []):
            name = self._get_name('identifier', item['name'])
            member_name = item.get('memberName', None)
            if member_name:
                member_name = self._get_name('attribute', member_name)
            identifiers.append(Identifier(name, member_name))
        return identifiers
