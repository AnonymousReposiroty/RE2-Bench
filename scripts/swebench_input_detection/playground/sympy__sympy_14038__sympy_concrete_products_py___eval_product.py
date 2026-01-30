# Problem: sympy__sympy-14038@@sympy.concrete.products.py@@_eval_product_L237
# Module: sympy.concrete.products
# Function: _eval_product
# Line: 237

from sympy import symbols, sympify, oo
from sympy.concrete.products import Product

# Create symbols
n = symbols('n')


def test_input(pred_input):
    # Ground truth values
    # term: 1 + n**(-2/3)
    # limits: ["1", "oo"] -> (n, 1, oo)
    # Note: limits format is (k, a, n) where k is variable, a is lower, n is upper

    term_gt = 1 + n**sympify('-2/3')
    # Input limits are ["1", "oo"], so full limits tuple is (n, 1, oo)
    limits_gt = (n, 1, oo)

    # Parse predicted inputs
    term_str = pred_input['args']['term']
    limits_list = pred_input['args']['limits']

    term_pred = sympify(term_str)
    # Reconstruct limits tuple: (n, limits[0], limits[1])
    # The variable n is inferred from the term
    limit_lower = sympify(limits_list[0])
    limit_upper = sympify(limits_list[1])
    limits_pred = (n, limit_lower, limit_upper)

    # Create Product instances and call _eval_product
    prod_gt = Product(term_gt, limits_gt)
    prod_pred = Product(term_pred, limits_pred)

    result_gt = prod_gt._eval_product(term_gt, limits_gt)
    result_pred = prod_pred._eval_product(term_pred, limits_pred)

    # Compare results
    assert result_gt == result_pred, f'Prediction failed! Expected: {result_gt}, Got: {result_pred}'


if __name__ == "__main__":
    # Test with the given input
    test_input_data = {
        "self": {},
        "args": {
            "term": "1 + n**(-2/3)",
            "limits": [
                "1",
                "oo"
            ]
        },
        "kwargs": {}
    }

    test_input(test_input_data)
    print("Test passed!")
