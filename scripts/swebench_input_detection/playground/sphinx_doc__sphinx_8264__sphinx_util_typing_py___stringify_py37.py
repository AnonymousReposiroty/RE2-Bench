# Problem: sphinx-doc__sphinx-8264@@sphinx.util.typing.py@@_stringify_py37_L89
# Module: sphinx.util.typing
# Function: _stringify_py37
# Line: 89

import sys

from sphinx.util.typing import _stringify_py37


# Create a class that mimics the annotation from input
class MyClass:
    """A test class to use as annotation."""
    pass


# Set attributes to match the input
MyClass.__module__ = 'test_util_typing'
MyClass.__doc__ = None


def test_input(pred_input):
    # Ground truth: a class with __module__='test_util_typing' and __doc__=None
    annotation_gt = MyClass

    # Create predicted annotation based on pred_input
    # The annotation should be a class/type with matching attributes
    class PredClass:
        pass

    pred_module = pred_input['args']['annotation'].get('__module__', None)
    pred_doc = pred_input['args']['annotation'].get('__doc__', None)

    PredClass.__module__ = pred_module
    PredClass.__doc__ = pred_doc
    # Set __qualname__ to match the ground truth class
    PredClass.__qualname__ = 'MyClass'

    # Call _stringify_py37 on both and compare results
    result_gt = _stringify_py37(annotation_gt)
    result_pred = _stringify_py37(PredClass)

    assert result_gt == result_pred, f'Prediction failed! Expected: {result_gt}, Got: {result_pred}'


if __name__ == "__main__":
    # Test with the given input
    test_input_data = {
        "self": {},
        "args": {
            "annotation": {
                "__module__": "test_util_typing",
                "__doc__": None
            }
        },
        "kwargs": {}
    }

    test_input(test_input_data)
    print("Test passed!")
