from warnings import warn
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.utils._bunch import Bunch

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

    def _check_warnings(self, *, params):
        """Check whether metadata is passed which is marked as WARN.

        If any metadata is passed which is marked as WARN, a warning is raised.

        Parameters
        ----------
        params : dict
            The metadata passed to a method.
        """
        params = {} if params is None else params
        warn_params = {prop for prop, alias in self._requests.items() if alias == WARN and prop in params}
        for param in warn_params:
            warn(f'Support for {param} has recently been added to {self.owner} class. To maintain backward compatibility, it is ignored now. Using `set_{self.method}_request({param}={{True, False}})` on this method of the class, you can set the request value to False to silence this warning, or to True to consume and use the metadata.')

    def _route_params(self, params, parent, caller):
        """Prepare the given metadata to be passed to the method.

        The output of this method can be used directly as the input to the
        corresponding method as **kwargs.

        Parameters
        ----------
        params : dict
            A dictionary of provided metadata.

        parent : object
            Parent class object, that routes the metadata.

        caller : str
            Method from the parent class object, where the metadata is routed from.

        Returns
        -------
        params : Bunch
            A :class:`~sklearn.utils.Bunch` of {metadata: value} which can be
            passed to the corresponding method.
        """
        self._check_warnings(params=params)
        unrequested = dict()
        args = {arg: value for arg, value in params.items() if value is not None}
        res = Bunch()
        for prop, alias in self._requests.items():
            if alias is False or alias == WARN:
                continue
            elif alias is True and prop in args:
                res[prop] = args[prop]
            elif alias is None and prop in args:
                unrequested[prop] = args[prop]
            elif alias in args:
                res[prop] = args[alias]
        if unrequested:
            if self.method in COMPOSITE_METHODS:
                callee_methods = COMPOSITE_METHODS[self.method]
            else:
                callee_methods = [self.method]
            set_requests_on = ''.join([f'.set_{method}_request({{metadata}}=True/False)' for method in callee_methods])
            message = f"[{', '.join([key for key in unrequested])}] are passed but are not explicitly set as requested or not requested for {_routing_repr(self.owner)}.{self.method}, which is used within {_routing_repr(parent)}.{caller}. Call `{_routing_repr(self.owner)}" + set_requests_on + '` for each metadata you want to request/ignore. See the Metadata Routing User guide <https://scikit-learn.org/stable/metadata_routing.html> for more information.'
            raise UnsetMetadataPassedError(message=message, unrequested_params=unrequested, routed_params=res)
        return res
