import copy
from botocore import xform_name
from botocore.utils import merge_dicts

class ResourceCollection:
    """
    Represents a collection of resources, which can be iterated through,
    optionally with filtering. Collections automatically handle pagination
    for you.

    See :ref:`guide_collections` for a high-level overview of collections,
    including when remote service requests are performed.

    :type model: :py:class:`~boto3.resources.model.Collection`
    :param model: Collection model
    :type parent: :py:class:`~boto3.resources.base.ServiceResource`
    :param parent: The collection's parent resource
    :type handler: :py:class:`~boto3.resources.response.ResourceHandler`
    :param handler: The resource response handler used to create resource
                    instances
    """

    def __init__(self, model, parent, handler, **kwargs):
        self._model = model
        self._parent = parent
        self._py_operation_name = xform_name(model.request.operation)
        self._handler = handler
        self._params = copy.deepcopy(kwargs)

    def _clone(self, **kwargs):
        """
        Create a clone of this collection. This is used by the methods
        below to provide a chainable interface that returns copies
        rather than the original. This allows things like:

            >>> base = collection.filter(Param1=1)
            >>> query1 = base.filter(Param2=2)
            >>> query2 = base.filter(Param3=3)
            >>> query1.params
            {'Param1': 1, 'Param2': 2}
            >>> query2.params
            {'Param1': 1, 'Param3': 3}

        :rtype: :py:class:`ResourceCollection`
        :return: A clone of this resource collection
        """
        params = copy.deepcopy(self._params)
        merge_dicts(params, kwargs, append_lists=True)
        clone = self.__class__(self._model, self._parent, self._handler, **params)
        return clone

    def all(self):
        """
        Get all items from the collection, optionally with a custom
        page size and item count limit.

        This method returns an iterable generator which yields
        individual resource instances. Example use::

            # Iterate through items
            >>> for queue in sqs.queues.all():
            ...     print(queue.url)
            'https://url1'
            'https://url2'

            # Convert to list
            >>> queues = list(sqs.queues.all())
            >>> len(queues)
            2
        """
        return self._clone()
