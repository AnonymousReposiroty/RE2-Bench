from __future__ import annotations
import inspect
from .conflict import ordering, ambiguities, super_signature, AmbiguityWarning
from .utils import expand_tuples

class Dispatcher:
    """ Dispatch methods based on type signature

    Use ``dispatch`` to add implementations

    Examples
    --------

    >>> from sympy.multipledispatch import dispatch
    >>> @dispatch(int)
    ... def f(x):
    ...     return x + 1

    >>> @dispatch(float)
    ... def f(x): # noqa: F811
    ...     return x - 1

    >>> f(3)
    4
    >>> f(3.0)
    2.0
    """
    __slots__ = ('__name__', 'name', 'funcs', 'ordering', '_cache', 'doc')

    def __init__(self, name, doc=None):
        self.name = self.__name__ = name
        self.funcs = {}
        self._cache = {}
        self.ordering = []
        self.doc = doc

    def register(self, *types, **kwargs):
        """ Register dispatcher with new implementation

        >>> from sympy.multipledispatch.dispatcher import Dispatcher
        >>> f = Dispatcher('f')
        >>> @f.register(int)
        ... def inc(x):
        ...     return x + 1

        >>> @f.register(float)
        ... def dec(x):
        ...     return x - 1

        >>> @f.register(list)
        ... @f.register(tuple)
        ... def reverse(x):
        ...     return x[::-1]

        >>> f(1)
        2

        >>> f(1.0)
        0.0

        >>> f([1, 2, 3])
        [3, 2, 1]
        """

        def _(func):
            self.add(types, func, **kwargs)
            return func
        return _

    @classmethod
    def get_func_params(cls, func):
        if hasattr(inspect, 'signature'):
            sig = inspect.signature(func)
            return sig.parameters.values()

    @classmethod
    def get_func_annotations(cls, func):
        """ Get annotations of function positional parameters
        """
        params = cls.get_func_params(func)
        if params:
            Parameter = inspect.Parameter
            params = (param for param in params if param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD))
            annotations = tuple((param.annotation for param in params))
            if not any((ann is Parameter.empty for ann in annotations)):
                return annotations

    def add(self, signature, func, on_ambiguity=ambiguity_warn):
        """ Add new types/method pair to dispatcher

        >>> from sympy.multipledispatch import Dispatcher
        >>> D = Dispatcher('add')
        >>> D.add((int, int), lambda x, y: x + y)
        >>> D.add((float, float), lambda x, y: x + y)

        >>> D(1, 2)
        3
        >>> D(1, 2.0)
        Traceback (most recent call last):
        ...
        NotImplementedError: Could not find signature for add: <int, float>

        When ``add`` detects a warning it calls the ``on_ambiguity`` callback
        with a dispatcher/itself, and a set of ambiguous type signature pairs
        as inputs.  See ``ambiguity_warn`` for an example.
        """
        if not signature:
            annotations = self.get_func_annotations(func)
            if annotations:
                signature = annotations
        if any((isinstance(typ, tuple) for typ in signature)):
            for typs in expand_tuples(signature):
                self.add(typs, func, on_ambiguity)
            return
        for typ in signature:
            if not isinstance(typ, type):
                str_sig = ', '.join((c.__name__ if isinstance(c, type) else str(c) for c in signature))
                raise TypeError('Tried to dispatch on non-type: %s\nIn signature: <%s>\nIn function: %s' % (typ, str_sig, self.name))
        self.funcs[signature] = func
        self.reorder(on_ambiguity=on_ambiguity)
        self._cache.clear()

    def reorder(self, on_ambiguity=ambiguity_warn):
        if _resolve[0]:
            self.ordering = ordering(self.funcs)
            amb = ambiguities(self.funcs)
            if amb:
                on_ambiguity(self, amb)
        else:
            _unresolved_dispatchers.add(self)
    __repr__ = __str__
