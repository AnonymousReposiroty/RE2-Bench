# Problem: sympy__sympy-13551@@sympy.concrete.products.py@@_eval_product_L236
# Module: sympy.concrete.products
# Function: _eval_product
# Line: 236

from sympy import symbols, sympify
from sympy.concrete.products import Product

# Create symbols
i, u, v = symbols('i u v')


def test_input(pred_input):
    # Ground truth values
    # term: (i**2 + 5*i + 4)/(i**2 + 2*i - 3)
    # limits: the variable is i, bounds from input are ["u + v", "u - 1"]
    # Note: limits format is (k, a, n) where k is variable, a is lower, n is upper

    term_gt = (i**2 + 5*i + 4)/(i**2 + 2*i - 3)
    # Input limits are [u + v, u - 1], so limits tuple is (i, u + v, u - 1)
    limits_gt = (i, u + v, u - 1)

    # Parse predicted inputs
    term_str = pred_input['args']['term']
    limits_list = pred_input['args']['limits']

    term_pred = sympify(term_str)
    # Reconstruct limits tuple: (i, limits[0], limits[1])
    # The variable i is inferred from the term
    limits_pred = (i, sympify(limits_list[0]), sympify(limits_list[1]))

    # Create Product instances and call _eval_product
    prod_gt = Product(term_gt, limits_gt)
    prod_pred = Product(term_pred, limits_pred)

    result_gt = prod_gt._eval_product(term_gt, limits_gt)
    result_pred = prod_pred._eval_product(term_pred, limits_pred)

    # Compare results (may need to simplify for comparison)
    assert result_gt == result_pred, f'Prediction failed! Expected: {result_gt}, Got: {result_pred}'


if __name__ == "__main__":
    # Test with the given input
    test_input_data = {
        "self": {},
        "args": {
            "term": "(i**2 + 5*i + 4)/(i**2 + 2*i - 3)",
            "limits": [
                "u + v",
                "u - 1"
            ]
        },
        "kwargs": {}
    }

    test_input(test_input_data)
    print("Test passed!")
