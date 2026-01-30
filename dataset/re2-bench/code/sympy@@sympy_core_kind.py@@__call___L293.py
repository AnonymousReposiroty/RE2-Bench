from .cache import cacheit
from sympy.multipledispatch.dispatcher import (Dispatcher,
    ambiguity_warn, ambiguity_register_error_ignore_dup,
    str_signature, RaiseNotImplementedError)

class KindDispatcher:
    """
    Dispatcher to select a kind from multiple kinds by binary dispatching.

    .. notes::
       This approach is experimental, and can be replaced or deleted in
       the future.

    Explanation
    ===========

    SymPy object's :obj:`sympy.core.kind.Kind()` vaguely represents the
    algebraic structure where the object belongs to. Therefore, with
    given operation, we can always find a dominating kind among the
    different kinds. This class selects the kind by recursive binary
    dispatching. If the result cannot be determined, ``UndefinedKind``
    is returned.

    Examples
    ========

    Multiplication between numbers return number.

    >>> from sympy import NumberKind, Mul
    >>> Mul._kind_dispatcher(NumberKind, NumberKind)
    NumberKind

    Multiplication between number and unknown-kind object returns unknown kind.

    >>> from sympy import UndefinedKind
    >>> Mul._kind_dispatcher(NumberKind, UndefinedKind)
    UndefinedKind

    Any number and order of kinds is allowed.

    >>> Mul._kind_dispatcher(UndefinedKind, NumberKind)
    UndefinedKind
    >>> Mul._kind_dispatcher(NumberKind, UndefinedKind, NumberKind)
    UndefinedKind

    Since matrix forms a vector space over scalar field, multiplication
    between matrix with numeric element and number returns matrix with
    numeric element.

    >>> from sympy.matrices import MatrixKind
    >>> Mul._kind_dispatcher(MatrixKind(NumberKind), NumberKind)
    MatrixKind(NumberKind)

    If a matrix with number element and another matrix with unknown-kind
    element are multiplied, we know that the result is matrix but the
    kind of its elements is unknown.

    >>> Mul._kind_dispatcher(MatrixKind(NumberKind), MatrixKind(UndefinedKind))
    MatrixKind(UndefinedKind)

    Parameters
    ==========

    name : str

    commutative : bool, optional
        If True, binary dispatch will be automatically registered in
        reversed order as well.

    doc : str, optional

    """

    def __init__(self, name, commutative=False, doc=None):
        self.name = name
        self.doc = doc
        self.commutative = commutative
        self._dispatcher = Dispatcher(name)

    def __call__(self, *args, **kwargs):
        if self.commutative:
            kinds = frozenset(args)
        else:
            kinds = []
            prev = None
            for a in args:
                if prev is not a:
                    kinds.append(a)
                    prev = a
        return self.dispatch_kinds(kinds, **kwargs)

    @cacheit
    def dispatch_kinds(self, kinds, **kwargs):
        if len(kinds) == 1:
            result, = kinds
            if not isinstance(result, Kind):
                raise RuntimeError('%s is not a kind.' % result)
            return result
        for i, kind in enumerate(kinds):
            if not isinstance(kind, Kind):
                raise RuntimeError('%s is not a kind.' % kind)
            if i == 0:
                result = kind
            else:
                prev_kind = result
                t1, t2 = (type(prev_kind), type(kind))
                k1, k2 = (prev_kind, kind)
                func = self._dispatcher.dispatch(t1, t2)
                if func is None and self.commutative:
                    func = self._dispatcher.dispatch(t2, t1)
                    k1, k2 = (k2, k1)
                if func is None:
                    result = UndefinedKind
                else:
                    result = func(k1, k2)
                if not isinstance(result, Kind):
                    raise RuntimeError('Dispatcher for {!r} and {!r} must return a Kind, but got {!r}'.format(prev_kind, kind, result))
        return result
