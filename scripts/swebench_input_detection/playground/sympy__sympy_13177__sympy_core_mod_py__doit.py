from sympy import ZeroMatrix, Identity, MatrixSymbol
from sympy import sympify
from sympy import ImmutableDenseMatrix
from validators.deserialize import deserialize
from sympy import sympify, Xor
from sympy import SparseMatrix, sympify
from sympy.core.singleton import S
from sympy.core.numbers import nan

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
        return f'dict_to_sparse_matrix({value})'
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
            return [sympify(i) for i in value]
    elif key == 'eq':
        if isinstance(value, str):
            return sympify({value})
        elif isinstance(value, dict):
            return f'reconstruct_boolean_from_log({value})'
    return deserialize(value)

def doit(p, q):
    if p.is_infinite or q.is_infinite or p is nan or (q is nan):
        return nan
    if p == q or p == -q or (p.is_Pow and p.exp.is_integer and (p.base == q) and q.is_integer and p.exp.is_positive) or (p.is_integer and q == 1):
        return S.Zero
    if q.is_Number:
        if p.is_Number:
            return p % q
        if q == 2:
            if p.is_even:
                return S.Zero
            elif p.is_odd:
                return S.One
    r = p / q
    try:
        d = int(r)
    except TypeError:
        pass
    else:
        if type(d) is int:
            rv = p - d * q
            if (rv * q < 0) == True:
                rv += q
            return rv
    d = p - q
    if d.is_negative:
        if q.is_negative:
            return d
        elif q.is_positive:
            return p


def test_input(pred_input):
	assert doit(p = deserialize_sympy('p', 'zoo'), q = deserialize_sympy('q', '0'))==doit(p = pred_input['args']['p'], q = pred_input['args']['q']), 'Prediction failed!'