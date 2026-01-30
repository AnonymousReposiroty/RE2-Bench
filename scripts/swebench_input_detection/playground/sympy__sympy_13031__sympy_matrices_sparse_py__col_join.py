from sympy import ZeroMatrix, Identity, MatrixSymbol
from sympy import sympify
from sympy import ImmutableDenseMatrix
from validators.deserialize import deserialize
from sympy import sympify, Xor
from sympy import SparseMatrix, sympify


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
            return f'sympify({value})'
        elif isinstance(value, dict):
            if not value:
                return value
            return f'dict_to_sympy_matrix({value})'
    elif key == 'coeffs' or key == 'args':
        if isinstance(value, list):
            return [sympify(i) for i in value]
    elif key == 'eq':
        if isinstance(value, str):
            return f'sympify({value})'
        elif isinstance(value, dict):
            return f'reconstruct_boolean_from_log({value})'
    return deserialize(value)


from sympy.matrices.sparse import MutableSparseMatrix

def test_input(pred_input):
	obj_ins = MutableSparseMatrix(_smat = deserialize_sympy('_smat', {}), rows = deserialize_sympy('rows', 0), cols = deserialize_sympy('cols', 0))
	obj_ins_pred = MutableSparseMatrix(_smat = pred_input['self']['_smat'], rows = pred_input['self']['rows'], cols = pred_input['self']['cols'])
	assert obj_ins.col_join(other = deserialize_sympy('other', {'rows': 2, 'cols': 1, '_mat': None}))==obj_ins_pred.col_join(other = pred_input['args']['other']), 'Prediction failed!'