import numpy as np
import ast
import re
from typing import Any, Dict

def convert_to_array(value: Any):

    # Already a numpy array: leave as is
    if isinstance(value, np.ndarray):
        return value

    # Case 4 & 5: Python list (1D or 2D or more)
    if isinstance(value, list):
        return np.array(value)

    # String-like inputs
    if isinstance(value, str):
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
            # Insert commas between rows: "][\n" -> "], ["
            s_norm = re.sub(r"\]\s*\n\s*\[", "], [", s)
            try:
                data = ast.literal_eval(s_norm)
                return np.array(data)
            except Exception:
                # Fallback: try loadtxt by stripping brackets
                import io
                try:
                    txt = s.replace("[", " ").replace("]", " ")
                    arr = np.loadtxt(io.StringIO(txt))
                    return arr
                except Exception:
                    return "invalid"

        # Case 2 (and general bracketed strings):
        #  - first try literal_eval for normal Python lists "[1, 2, 3]"
        #  - otherwise parse numbers via np.fromstring for "[ 2  4  6 ... ]"
        if s.startswith("[") and s.endswith("]"):
            # Try: Python-style list
            try:
                data = ast.literal_eval(s)
                return np.array(data)
            except Exception:
                # Try: space-separated numbers without commas
                arr = np.fromstring(s.strip("[]"), sep=" ")
                if arr.size > 0:
                    return arr
                return "invalid"

    # For any other type, just return as is
    return value


def convert_dict_values_to_arrays(d: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Visit each key in the dictionary and convert its value
    into a numpy array (or 'invalid') according to the rules above.
    """
    out = {}
    for k, v in d.items():
        out[k] = convert_to_array(v)
    return out

def test():
    example = {
    "case1": "[[ 0.02090526 -0.02984846  0.04445676  0.00073659 -0.03625016]\n"
             " [-0.02984846  0.15811976 -0.10110064 -0.14692452  0.11975385]\n"
             " [ 0.04445676 -0.10110064  0.57855486 -0.18284347 -0.33906752]\n"
             " [ 0.00073659 -0.14692452 -0.18284347  0.6706584  -0.341627  ]\n"
             " [-0.03625016  0.11975385 -0.33906752 -0.341627    0.59719083]]",
    "case2": "[ 2  4  6  8 10 12 14 16 18]",
    "case3": "array([1, 1, 2, 2])",
    "case4": [0, 0, 0, 0, 0, 0, 0],
    "case5": [[0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]],
    "case6": "[[0.090962, -0.12025564, -0.02202145], [0.090962, -0.12025564, -0.02202145], ..., [0.090962, -0.12025564, -0.02202145]]",
            }

    converted = convert_dict_values_to_arrays(example)
    for k, v in converted.items():
        print(k, "->", type(v), "\n", v, "\n")