from typing import Generic, Protocol, Any
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.domains.domain import Domain, Er, Es
from sympy.polys.orderings import lex, MonomialOrder
from sympy.polys.rings import PolyRing, PolyElement
from sympy.printing.defaults import DefaultPrinting

class FracField(DefaultPrinting, Generic[Er]):
    """Multivariate distributed rational function field. """
    ring: PolyRing[Er]
    gens: tuple[FracElement[Er], ...]
    symbols: tuple[Expr, ...]
    ngens: int
    domain: Domain[Er]
    order: MonomialOrder
    zero: FracElement[Er]
    one: FracElement[Er]
    dtype: FracElementConstructor[Er]
    _hash: int
    _hash_tuple: Any

    def __new__(cls, symbols, domain: Domain[Er], order: str | MonomialOrder | None=lex) -> FracField[Er]:
        ring = PolyRing(symbols, domain, order)
        symbols = ring.symbols
        ngens = ring.ngens
        domain = ring.domain
        order = ring.order
        _hash_tuple = (cls.__name__, symbols, ngens, domain, order)
        obj = object.__new__(cls)
        obj._hash_tuple = _hash_tuple
        obj._hash = hash(_hash_tuple)
        obj.ring = ring
        obj.symbols = symbols
        obj.ngens = ngens
        obj.domain = domain
        obj.order = order
        obj.dtype = FracElement(obj, ring.zero).raw_new
        obj.zero = obj.dtype(ring.zero)
        obj.one = obj.dtype(ring.one)
        obj.gens = obj._gens()
        for symbol, generator in zip(obj.symbols, obj.gens):
            if isinstance(symbol, Symbol):
                name = symbol.name
                if not hasattr(obj, name):
                    setattr(obj, name, generator)
        return obj
    __call__ = field_new
