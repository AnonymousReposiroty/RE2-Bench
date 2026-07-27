from sympy.core.singleton import Singleton, S

class BooleanFalse(BooleanAtom, metaclass=Singleton):
    """
    SymPy version of ``False``, a singleton that can be accessed via ``S.false``.

    This is the SymPy version of ``False``, for use in the logic module. The
    primary advantage of using ``false`` instead of ``False`` is that shorthand
    Boolean operations like ``~`` and ``>>`` will work as expected on this class,
    whereas with ``False`` they act bitwise on 0. Functions in the logic module
    will return this class when they evaluate to false.

    Notes
    ======

    See the notes section in :py:class:`sympy.logic.boolalg.BooleanTrue`

    Examples
    ========

    >>> from sympy import sympify, true, false, Or
    >>> sympify(False)
    False
    >>> _ is False, _ is false
    (False, True)

    >>> Or(true, false)
    True
    >>> _ is true
    True

    Python operators give a boolean result for false but a
    bitwise result for False

    >>> ~false, ~False  # doctest: +SKIP
    (True, -1)
    >>> false >> false, False >> False
    (True, 0)

    See Also
    ========

    sympy.logic.boolalg.BooleanTrue

    """

    def __eq__(self, other):
        if other is True:
            return False
        if other is False:
            return True
        return super().__eq__(other)
