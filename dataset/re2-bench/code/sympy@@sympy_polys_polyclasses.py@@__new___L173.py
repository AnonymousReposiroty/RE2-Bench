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
from sympy.polys.polyerrors import (
    CoercionFailed,
    ExactQuotientFailed,
    DomainError,
    NotInvertible,
)
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
import flint

class DMP(CantSympify, Generic[Er]):
    """Dense Multivariate Polynomials over `K`. """
    __slots__ = ()
    lev: int
    dom: Domain[Er]

    def __new__(cls, rep: dmp[Er], dom: Domain[Er], lev: int | None=None):
        if lev is None:
            rep, lev = dmp_validate(rep, dom)
        elif not isinstance(rep, list):
            raise CoercionFailed('expected list, got %s' % type(rep))
        return cls.new(rep, dom, lev)

    @classmethod
    def new(cls, rep: dmp[Er], dom: Domain[Er], lev: int) -> DMP_Python[Er] | DUP_Flint[Er]:
        if flint is not None:
            if lev == 0 and _supported_flint_domain(dom):
                return DUP_Flint._new(rep, dom, lev)
        return DMP_Python._new(rep, dom, lev)
