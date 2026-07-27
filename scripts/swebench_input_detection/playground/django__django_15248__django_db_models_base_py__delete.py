# Problem: django__django-15248@@django.db.models.base.py@@delete_L983
# Module: django.db.models.base
# Function: delete
# Line: 983

import sys
import os

# Add workspace to path for the specific django version

import django
from django.conf import settings

# Configure Django settings if not already configured
if not settings.configured:
    settings.configure(
        DEBUG=True,
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': ':memory:',
            }
        },
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'django.contrib.auth',
        ],
        USE_TZ=True,
        DEFAULT_AUTO_FIELD='django.db.models.AutoField',
    )
    django.setup()

from django.db import models, connection


# Define a simple model matching the input structure
class Person(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)

    class Meta:
        app_label = 'test_app'


def test_input(pred_input):
    # Ground truth:
    # self: Person instance with id=99998, first_name="James", last_name="Jones"
    # args: {} (empty)
    # kwargs: {} (empty - uses defaults: using=None, keep_parents=False)

    # Create the table
    with connection.schema_editor() as schema_editor:
        try:
            schema_editor.create_model(Person)
        except Exception:
            pass  # Table might already exist

    # Parse predicted input
    self_data = pred_input['self']
    using_pred = pred_input['kwargs'].get('using', None)
    keep_parents_pred = pred_input['kwargs'].get('keep_parents', False)

    # Ground truth values
    id_gt = 99998
    first_name_gt = "James"
    last_name_gt = "Jones"
    using_gt = None
    keep_parents_gt = False

    # Test 1: Create and delete ground truth instance
    person_gt = Person(id=id_gt, first_name=first_name_gt, last_name=last_name_gt)
    person_gt._state.db = "default"
    person_gt._state.adding = False
    person_gt.save()

    result_gt = person_gt.delete(using=using_gt, keep_parents=keep_parents_gt)

    # Check the object was deleted
    assert not Person.objects.filter(id=id_gt).exists(), "Ground truth delete failed"

    # Test 2: Create and delete predicted instance (with different ID to avoid collision)
    pred_id = self_data.get('id', 99998)
    # Use a different ID for predicted to avoid collision if IDs are same
    actual_pred_id = pred_id + 1 if pred_id == id_gt else pred_id

    person_pred = Person(
        id=actual_pred_id,
        first_name=self_data.get('first_name', 'James'),
        last_name=self_data.get('last_name', 'Jones')
    )
    person_pred._state.db = self_data.get('_state', {}).get('db', 'default')
    person_pred._state.adding = self_data.get('_state', {}).get('adding', False)
    person_pred.save()

    result_pred = person_pred.delete(using=using_pred, keep_parents=keep_parents_pred)

    # Compare results: both should return tuple (count, {model: count})
    # The delete count should be the same (1 object deleted)
    assert result_gt[0] == result_pred[0], f'Prediction failed! Expected count: {result_gt[0]}, Got: {result_pred[0]}'


if __name__ == "__main__":
    # Test with the given input
    test_input_data = {
        "self": {
            "_state": {
                "db": "default",
                "adding": False
            },
            "id": 99998,
            "first_name": "James",
            "last_name": "Jones"
        },
        "args": {},
        "kwargs": {}
    }

    test_input(test_input_data)
    print("Test passed!")
