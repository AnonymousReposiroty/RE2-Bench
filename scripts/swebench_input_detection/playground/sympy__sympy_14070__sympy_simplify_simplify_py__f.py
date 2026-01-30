from sympy import ZeroMatrix, Identity, MatrixSymbol
from sympy import sympify
from sympy import ImmutableDenseMatrix
from validators.deserialize import deserialize
from sympy import sympify, Xor
from sympy import SparseMatrix, sympify
from sympy.parsing.sympy_parser import parse_expr

def reconstruct_boolean_from_log(d):
    if "_argset" not in d:
        raise ValueError("Missing _argset; cannot reconstruct expression")
    raw = d["_argset"].strip()
    if not (raw.startswith("frozenset({") and raw.endswith("})")):
        raise ValueError(f"Unexpected format: {raw}")
    inner = raw[len("frozenset({"):-2].strip()
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    args = [sympify(p) for p in parts]
    return Xor(*args)


def dict_to_sympy_matrix(d, name="X"):
    rows = d.get("_rows")
    cols = d.get("_cols")
    mat = d.get("_mat")
    if mat is None:
        if rows == cols:
            return Identity(rows)
        else:
            return MatrixSymbol(name, rows, cols)
    else:
        return ImmutableDenseMatrix(mat)


def dict_to_sparse_matrix(log_dict):
    if not all(k in log_dict for k in ('_smat', 'rows', 'cols')):
        raise ValueError("Dictionary must have '_smat', 'rows', and 'cols' keys")
    smat_dict = log_dict['_smat']
    rows, cols = log_dict['rows'], log_dict['cols']
    entries = {}
    for key_str, val_str in smat_dict.items():
        r, c = map(int, key_str.strip('()').split(','))
        entries[r, c] = sympify(val_str)

    return SparseMatrix(rows, cols, entries)


def deserialize_sympy(key, value):
    if isinstance(value, dict) and 'cols' in value.keys() and 'rows' in value.keys() and '_smat' in value.keys():
        return dict_to_sparse_matrix({value})
    if key == 'expr' or key == 'rv' or key == 'p' or key == 'q' or key == 'e' or key == 'term':
        if isinstance(value, str):
            value = value.replace('"', '')
            return sympify({value})
        elif isinstance(value, dict):
            if not value:
                return value
            return dict_to_sympy_matrix({value})
    elif key == 'coeffs' or key == 'args':
        if isinstance(value, list):
            return [parse_expr(i) for i in value]
    elif key == 'eq':
        if isinstance(value, str):
            return parse_expr({value})
        elif isinstance(value, dict):
            return reconstruct_boolean_from_log({value})
    return deserialize(value)

from sympy.core import (S,Mul)
from sympy.functions import sin, cos, exp, cosh, tanh, sinh, tan, cot, coth

def f(rv):
    if not rv.is_Mul:
        return rv
    commutative_part, noncommutative_part = rv.args_cnc()
    if (len(noncommutative_part) > 1):
        return f(Mul(*commutative_part))*Mul(*noncommutative_part)
    rvd = rv.as_powers_dict()
    newd = rvd.copy()

    def signlog(expr, sign=S.One):
        if expr is S.Exp1:
            return sign, S.One
        elif isinstance(expr, exp) or (expr.is_Pow and expr.base == S.Exp1):
            return sign, expr.exp
        elif sign is S.One:
            return signlog(-expr, sign=-S.One)
        else:
            return None, None

    ee = rvd[S.Exp1]
    for k in rvd:
        if k.is_Add and len(k.args) == 2:
            # k == c*(1 + sign*E**x)
            c = k.args[0]
            sign, x = signlog(k.args[1]/c)
            if not x:
                continue
            m = rvd[k]
            newd[k] -= m
            if ee == -x*m/2:
                newd[S.Exp1] -= ee
                ee = 0
                if sign == 1:
                    newd[2*c*cosh(x/2)] += m
                else:
                    newd[-2*c*sinh(x/2)] += m
            elif newd[1 - sign*S.Exp1**x] == -m:
                # tanh
                del newd[1 - sign*S.Exp1**x]
                if sign == 1:
                    newd[-c/tanh(x/2)] += m
                else:
                    newd[-c*tanh(x/2)] += m
            else:
                newd[1 + sign*S.Exp1**x] += m
                newd[c] += m

    return Mul(*[k**newd[k] for k in newd])

def test_input(pred_input):
	assert f(rv = deserialize_sympy('rv', '-"log(2)" + log(3)'))==f(rv = pred_input['args']['rv']), 'Prediction failed!'
