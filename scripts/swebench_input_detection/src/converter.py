import numpy as np
import ast
import re
from typing import Any, Dict

from sympy import sympify
from sympy import SparseMatrix, sympify
from sympy import ImmutableDenseMatrix
from sympy import sympify, Xor
from sympy import ZeroMatrix, Identity, MatrixSymbol
from sympy.parsing.sympy_parser import parse_expr

class SklearnConverter:
    def convert_scalar_to_array(self,value: Any):
        """
        Handle the string / scalar cases only (cases 1–3, 6).
        Lists and dicts are handled by the recursive wrapper.
        """

        # Already a numpy array: leave as is
        if isinstance(value, np.ndarray):
            return value

        # Non-string scalars: just return as is
        if not isinstance(value, str):
            return value

        s = value.strip()

        # Case 6: contains "..."
        if "..." in s:
            return "invalid"

        # Case 3: "array([1, 1, 2, 2])"
        if s.startswith("array(") and s.endswith(")"):
            inner = s[len("array("):-1]
            try:
                data = ast.literal_eval(inner)
                return np.array(data)
            except Exception:
                return "invalid"

        # Case 1: 2D matrix with newlines like "[[...]\n [...]\n ...]"
        if s.startswith("[[") and "\n" in s:
            # Normalize "][\n" -> "], ["
            s_norm = re.sub(r"\]\s*\n\s*\[", "], [", s)
            try:
                data = ast.literal_eval(s_norm)
                return np.array(data)
            except Exception:
                # Fallback: parse as plain numbers
                try:
                    txt = s.replace("[", " ").replace("]", " ")
                    arr = np.loadtxt(io.StringIO(txt))
                    return arr
                except Exception:
                    return "invalid"

        # Case 2 (and generic bracketed numeric strings)
        if s.startswith("[") and s.endswith("]"):
            # First try Python-style list
            try:
                data = ast.literal_eval(s)
                return np.array(data)
            except Exception:
                # Then try space-separated numbers
                arr = np.fromstring(s.strip("[]"), sep=" ")
                if arr.size > 0:
                    return arr
                return "invalid"

        # Anything else: leave unchanged
        return value


    def convert_structure(self, obj: Any) -> Any:
        """
        Recursively walk dicts/lists/tuples and convert values
        into numpy arrays (or 'invalid') when they match the patterns.
        """

        # Dict: recurse on values
        if isinstance(obj, dict):
            return {k: self.convert_structure(v) for k, v in obj.items()}

        # List or tuple: decide whether to convert whole list to np.array,
        # or recurse into elements.
        if isinstance(obj, (list, tuple)):
            # Try to see if this can be a "nice" numeric/str array
            try:
                arr = np.array(obj)
                # If dtype is not object, we assume it's numeric/str-like -> convert
                if arr.dtype != object:
                    return arr
            except Exception:
                pass

            # Otherwise, recurse element-wise and keep as list
            return [self.convert_structure(v) for v in obj]

        # Everything else: handle scalar/string cases
        return self.convert_scalar_to_array(obj)


    def convert(self, d: Dict[Any, Any]) -> Dict[Any, Any]:
        return self.convert_structure(d)
    
    



class SympyConverter:
    def reconstruct_boolean_from_log(self, d):
        if "_argset" not in d:
            raise ValueError("Missing _argset; cannot reconstruct expression")
        raw = d["_argset"].strip()
        if not (raw.startswith("frozenset({") and raw.endswith("})")):
            raise ValueError(f"Unexpected format: {raw}")
        inner = raw[len("frozenset({"):-2].strip()
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        args = [sympify(p) for p in parts]
        return Xor(*args)
    
    def dict_to_sparse_matrix(self, log_dict):
        if not all(k in log_dict for k in ('_smat', 'rows', 'cols')):
            raise ValueError("Dictionary must have '_smat', 'rows', and 'cols' keys")
        smat_dict = log_dict['_smat']
        rows, cols = log_dict['rows'], log_dict['cols']
        entries = {}
        for key_str, val_str in smat_dict.items():
            r, c = map(int, key_str.strip('()').split(','))
            entries[r, c] = sympify(val_str)

        return SparseMatrix(rows, cols, entries)


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
        
    def deserialize_sympy(self, key, value):
        # sparse matrix case
        if isinstance(value, dict) and 'cols' in value and 'rows' in value and '_smat' in value:
            return f'dict_to_sparse_matrix({value})'

        # expression-like keys
        if key in ('expr', 'rv', 'p', 'q', 'e', 'term'):
            if isinstance(value, str):
                value = value.replace('"', '')
                return sympify({value})
            elif isinstance(value, dict):
                if not value:
                    return value
                return self.dict_to_sympy_matrix({value})

        # list of coeffs/args
        elif key in ('coeffs', 'args'):
            if isinstance(value, list):
                return [sympify(i) for i in value]

        # equations
        elif key == 'eq':
            if isinstance(value, str):
                return sympify({value})
            elif isinstance(value, dict):
                return self.reconstruct_boolean_from_log({value})
        return value


    def convert(self, x):
        # Recurse into containers first
        if isinstance(x, dict):
            new_d = {}
            for k, v in x.items():
                # First recursively convert nested structures
                converted_v = self.convert(v)
                # Then apply the key-specific deserialization logic
                new_d[k] = self.deserialize_sympy(k, converted_v)
            return new_d

        elif isinstance(x, list):
            return [self.convert(i) for i in x]

        elif isinstance(x, tuple):
            return tuple(self.convert(i) for i in x)

        # Base case: non-container
        return x