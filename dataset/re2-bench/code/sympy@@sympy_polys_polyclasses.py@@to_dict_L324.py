from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    overload,
    Callable,
    TypeVar,
)
from sympy.core.sympify import CantSympify
from sympy.polys.domains import Domain, ZZ, QQ
from sympy.polys.domains.domain import Er, Es, Et, Eg
from sympy.polys.densebasic import (
    _dup,
    _dmp,
    dmp,
    dmp_tup,
    monom,
    dmp_validate,
    dup_normal, dmp_normal,
    dup_convert, dmp_convert,
    dmp_from_sympy,
    dup_strip,
    dmp_degree_in,
    dmp_degree_list,
    dmp_negative_p,
    dmp_ground_LC,
    dmp_ground_TC,
    dmp_ground_nth,
    dmp_one, dmp_ground,
    dmp_zero, dmp_zero_p, dmp_one_p, dmp_ground_p,
    dup_from_dict, dup_from_raw_dict, dmp_from_dict,
    dmp_to_dict,
    dmp_deflate,
    dmp_inject, dmp_eject,
    dmp_terms_gcd,
    dmp_list_terms, dmp_exclude,
    dup_slice, dmp_slice_in, dmp_permute,
    dmp_to_tuple,)

class DMP(CantSympify, Generic[Er]):
    """Dense Multivariate Polynomials over `K`. """
    __slots__ = ()
    lev: int
    dom: Domain[Er]

    def to_dict(f, zero: bool=False) -> dict[monom, Er]:
        """Convert ``f`` to a dict representation with native coefficients. """
        if zero and (not f):
            return {(0,) * (f.lev + 1): f.dom.zero}
        else:
            return dmp_to_dict(f.to_list(), f.lev, f.dom)
