from functools import partial
from sympy.strategies import chain, minimize
from sympy.strategies.core import identity

def greedy(tree, objective=identity, **kwargs):
    """ Execute a strategic tree.  Select alternatives greedily

    Trees
    -----

    Nodes in a tree can be either

    function - a leaf
    list     - a selection among operations
    tuple    - a sequence of chained operations

    Textual examples
    ----------------

    Text: Run f, then run g, e.g. ``lambda x: g(f(x))``
    Code: ``(f, g)``

    Text: Run either f or g, whichever minimizes the objective
    Code: ``[f, g]``

    Textx: Run either f or g, whichever is better, then run h
    Code: ``([f, g], h)``

    Text: Either expand then simplify or try factor then foosimp. Finally print
    Code: ``([(expand, simplify), (factor, foosimp)], print)``

    Objective
    ---------

    "Better" is determined by the objective keyword.  This function makes
    choices to minimize the objective.  It defaults to the identity.

    Examples
    ========

    >>> from sympy.strategies.tree import greedy
    >>> inc    = lambda x: x + 1
    >>> dec    = lambda x: x - 1
    >>> double = lambda x: 2*x

    >>> tree = [inc, (dec, double)] # either inc or dec-then-double
    >>> fn = greedy(tree)
    >>> fn(4)  # lowest value comes from the inc
    5
    >>> fn(1)  # lowest value comes from dec then double
    0

    This function selects between options in a tuple.  The result is chosen
    that minimizes the objective function.

    >>> fn = greedy(tree, objective=lambda x: -x)  # maximize
    >>> fn(4)  # highest value comes from the dec then double
    6
    >>> fn(1)  # highest value comes from the inc
    2

    Greediness
    ----------

    This is a greedy algorithm.  In the example:

        ([a, b], c)  # do either a or b, then do c

    the choice between running ``a`` or ``b`` is made without foresight to c
    """
    optimize = partial(minimize, objective=objective)
    return treeapply(tree, {list: optimize, tuple: chain}, **kwargs)
