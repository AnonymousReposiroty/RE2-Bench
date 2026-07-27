from .add import Add
from .basic import Basic
from .singleton import S
from .containers import Tuple, Dict

def _gcd_terms(terms, isprimitive=False, fraction=True):
    """Helper function for :func:`gcd_terms`.

    Parameters
    ==========

    isprimitive : boolean, optional
        If ``isprimitive`` is True then the call to primitive
        for an Add will be skipped. This is useful when the
        content has already been extracted.

    fraction : boolean, optional
        If ``fraction`` is True then the expression will appear over a common
        denominator, the lcm of all term denominators.
    """

    if isinstance(terms, Basic) and not isinstance(terms, Tuple):
        terms = Add.make_args(terms)

    terms = list(map(Term, [t for t in terms if t]))

    # there is some simplification that may happen if we leave this
    # here rather than duplicate it before the mapping of Term onto
    # the terms
    if len(terms) == 0:
        return S.Zero, S.Zero, S.One

    if len(terms) == 1:
        cont = terms[0].coeff
        numer = terms[0].numer.as_expr()
        denom = terms[0].denom.as_expr()

    else:
        cont = terms[0]
        for term in terms[1:]:
            cont = cont.gcd(term)

        for i, term in enumerate(terms):
            terms[i] = term.quo(cont)

        if fraction:
            denom = terms[0].denom

            for term in terms[1:]:
                denom = denom.lcm(term.denom)

            numers = []
            for term in terms:
                numer = term.numer.mul(denom.quo(term.denom))
                numers.append(term.coeff*numer.as_expr())
        else:
            numers = [t.as_expr() for t in terms]
            denom = Term(S.One).numer

        cont = cont.as_expr()
        numer = Add(*numers)
        denom = denom.as_expr()

    if not isprimitive and numer.is_Add:
        _cont, numer = numer.primitive()
        cont *= _cont

    return cont, numer, denom
