from sympy.core.symbol import Dummy, Symbol
from sympy.polys.constructor import construct_domain
from sympy.polys.polyclasses import DMP, DMF, ANP

def named_poly(n, f, K, name, x, polys):
    r"""Common interface to the low-level polynomial generating functions
    in orthopolys and appellseqs.

    Parameters
    ==========

    n : int
        Index of the polynomial, which may or may not equal its degree.
    f : callable
        Low-level generating function to use.
    K : Domain or None
        Domain in which to perform the computations. If None, use the smallest
        field containing the rationals and the extra parameters of x (see below).
    name : str
        Name of an arbitrary individual polynomial in the sequence generated
        by f, only used in the error message for invalid n.
    x : seq
        The first element of this argument is the main variable of all
        polynomials in this sequence. Any further elements are extra
        parameters required by f.
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    if n < 0:
        raise ValueError("Cannot generate %s of index %s" % (name, n))
    head, tail = x[0], x[1:]
    if K is None:
        K, tail = construct_domain(tail, field=True)
    poly = DMP(f(int(n), *tail, K), K)
    if head is None:
        poly = PurePoly.new(poly, Dummy('x'))
    else:
        poly = Poly.new(poly, head)
    return poly if polys else poly.as_expr()
