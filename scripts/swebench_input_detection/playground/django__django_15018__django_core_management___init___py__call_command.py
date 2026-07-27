# Problem: django__django-15018@@django.core.management.__init__.py@@call_command_L78
# Module: django.core.management
# Function: call_command
# Line: 78

import sys
import io

# Add workspace to path for the specific django version

import django
from django.conf import settings

# Configure Django settings if not already configured
if not settings.configured:
    settings.configure(
        DEBUG=True,
        DATABASES={},
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'django.contrib.auth',
        ],
        USE_TZ=True,
    )
    django.setup()

from django.core.management import call_command


def test_input(pred_input):
    # Ground truth:
    # command_name: "transaction"
    # stdout: {} in input is a placeholder - should be a file-like object
    # no_color: True

    # Parse predicted inputs
    command_name_pred = pred_input['args']['command_name']
    no_color_pred = pred_input['kwargs'].get('no_color', False)

    # For stdout, if it's empty dict {}, use StringIO
    stdout_dict = pred_input['kwargs'].get('stdout', {})
    if isinstance(stdout_dict, dict) and not stdout_dict:
        stdout_pred = io.StringIO()
    else:
        stdout_pred = io.StringIO()

    # Ground truth values
    command_name_gt = "transaction"
    no_color_gt = True
    stdout_gt = io.StringIO()

    # Both ground truth and predicted command "transaction" don't exist
    # so we compare if they raise the same error
    gt_error = None
    pred_error = None

    try:
        call_command(command_name_gt, stdout=stdout_gt, no_color=no_color_gt)
    except Exception as e:
        gt_error = type(e).__name__

    try:
        call_command(command_name_pred, stdout=stdout_pred, no_color=no_color_pred)
    except Exception as e:
        pred_error = type(e).__name__

    # Compare: both should raise same type of error for unknown command
    assert gt_error == pred_error, f'Prediction failed! Expected error: {gt_error}, Got: {pred_error}'


if __name__ == "__main__":
    # Test with the given input
    test_input_data = {
        "self": {},
        "args": {
            "command_name": "transaction"
        },
        "kwargs": {
            "stdout": {},
            "no_color": True
        }
    }

    test_input(test_input_data)
    print("Test passed!")
