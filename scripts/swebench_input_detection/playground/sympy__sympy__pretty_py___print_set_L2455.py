# Problem: sympy@@sympy__printing_pretty_pretty.py@@_print_set_L2455
# Module: sympy.printing.pretty.pretty
# Function: _print_set
# Line: 2455

from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.assumptions.ask import Q


def create_printer(settings=None, print_level=1):
    """
    Create a PrettyPrinter instance with the given settings.
    """
    printer = PrettyPrinter(settings)
    printer._print_level = print_level
    return printer


def test_input(pred_input):
    # Create ground truth printer with default settings
    settings_gt = {
        "order": None,
        "full_prec": "auto",
        "use_unicode": None,
        "wrap_line": True,
        "num_columns": None,
        "use_unicode_sqrt_char": True,
        "root_notation": True,
        "mat_symbol_style": "plain",
        "imaginary_unit": "i",
        "perm_cyclic": True
    }
    printer_gt = create_printer(settings_gt, print_level=1)

    # Create ground truth set: {Q.integer, Q.positive}
    s_gt = {Q.integer, Q.positive}

    # Create predicted printer with settings from pred_input
    settings_pred = {}
    if pred_input['self'].get('_settings'):
        for key, val in pred_input['self']['_settings'].items():
            # Convert 'null' strings to None
            if val == 'null' or val is None:
                settings_pred[key] = None
            else:
                settings_pred[key] = val

    print_level_pred = pred_input['self'].get('_print_level', 1)
    printer_pred = create_printer(settings_pred, print_level=print_level_pred)

    # Create predicted set from args
    s_pred_items = pred_input['args']['s']
    s_pred = set()
    for item in s_pred_items:
        # Map string representation to actual Q predicates
        if item == "Q.integer":
            s_pred.add(Q.integer)
        elif item == "Q.positive":
            s_pred.add(Q.positive)
        elif item == "Q.negative":
            s_pred.add(Q.negative)
        elif item == "Q.real":
            s_pred.add(Q.real)
        elif item == "Q.even":
            s_pred.add(Q.even)
        elif item == "Q.odd":
            s_pred.add(Q.odd)
        elif item == "Q.prime":
            s_pred.add(Q.prime)
        elif item == "Q.composite":
            s_pred.add(Q.composite)
        elif item == "Q.zero":
            s_pred.add(Q.zero)
        elif item == "Q.nonzero":
            s_pred.add(Q.nonzero)
        elif item == "Q.rational":
            s_pred.add(Q.rational)
        elif item == "Q.irrational":
            s_pred.add(Q.irrational)
        elif item == "Q.finite":
            s_pred.add(Q.finite)
        elif item == "Q.infinite":
            s_pred.add(Q.infinite)
        elif item == "Q.commutative":
            s_pred.add(Q.commutative)
        else:
            # Try to get it dynamically from Q
            attr_name = item.replace("Q.", "")
            if hasattr(Q, attr_name):
                s_pred.add(getattr(Q, attr_name))
            else:
                raise ValueError(f"Unknown predicate: {item}")

    # Call _print_set on both and compare results
    result_gt = printer_gt._print_set(s_gt)
    result_pred = printer_pred._print_set(s_pred)

    # Compare the rendered output
    assert str(result_gt) == str(result_pred), f'Prediction failed! Expected: {result_gt}, Got: {result_pred}'


if __name__ == "__main__":
    # Test with the given input
    test_input_data = {
        "self": {
            "_str": "<class 'str'>",
            "_settings": {
                "order": None,
                "full_prec": "auto",
                "use_unicode": None,
                "wrap_line": True,
                "num_columns": None,
                "use_unicode_sqrt_char": True,
                "root_notation": True,
                "mat_symbol_style": "plain",
                "imaginary_unit": "i",
                "perm_cyclic": True
            },
            "_context": {},
            "_print_level": 1
        },
        "args": {
            "s": [
                "Q.integer",
                "Q.positive"
            ]
        },
        "kwargs": {}
    }

    test_input(test_input_data)
    print("Test passed!")
