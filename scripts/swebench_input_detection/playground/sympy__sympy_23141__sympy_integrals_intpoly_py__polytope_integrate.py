# Problem: sympy__sympy-23141@@sympy.integrals.intpoly.py@@polytope_integrate_L28
# Module: sympy.integrals.intpoly
# Function: polytope_integrate
# Line: 28

from sympy import symbols, sympify, Point, Polygon
from sympy.integrals.intpoly import polytope_integrate

# Create symbols
x, y = symbols('x y')


def test_input(pred_input):
    # Ground truth:
    # poly: {} in input is a placeholder - clockwise=True requires a Polygon
    # expr: x*y
    # clockwise: True

    # Create a simple polygon (unit square) for testing
    # When clockwise=True, the function sorts vertices clockwise
    poly_gt = Polygon(Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))
    expr_gt = x * y

    # Parse predicted inputs
    expr_str = pred_input['args']['expr']
    expr_pred = sympify(expr_str)
    clockwise_pred = pred_input['kwargs'].get('clockwise', False)

    # For poly, if it's empty dict {}, use the same polygon as ground truth
    poly_dict = pred_input['args']['poly']
    if isinstance(poly_dict, dict) and not poly_dict:
        # Empty dict - use default polygon
        poly_pred = poly_gt
    else:
        poly_pred = poly_gt  # Use same polygon for comparison

    # Call polytope_integrate on both
    result_gt = polytope_integrate(poly_gt, expr_gt, clockwise=True)
    result_pred = polytope_integrate(poly_pred, expr_pred, clockwise=clockwise_pred)

    # Compare results
    assert result_gt == result_pred, f'Prediction failed! Expected: {result_gt}, Got: {result_pred}'


if __name__ == "__main__":
    # Test with the given input
    test_input_data = {
        "self": {},
        "args": {
            "poly": {},
            "expr": "x*y"
        },
        "kwargs": {
            "clockwise": True
        }
    }

    test_input(test_input_data)
    print("Test passed!")
