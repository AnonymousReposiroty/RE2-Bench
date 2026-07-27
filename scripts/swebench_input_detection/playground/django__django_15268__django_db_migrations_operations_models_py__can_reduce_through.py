# Problem: django__django-15268@@django.db.migrations.operations.models.py@@can_reduce_through
# Benchmark: Swebench
# Module: django.db.migrations.operations.models
# Function: can_reduce_through

from django.db.migrations.operations.models import AlterTogetherOptionOperation


def test_input(pred_input):
    assert {'_constructor_args': [[], {'name': 'book', 'index_together': "{('title', 'newfield2')}"}], 'index_together': "{('title', 'newfield2')}", 'name': 'book', '_auto_deps': None, 'name_lower': 'book', 'option_value': "{('title', 'newfield2')}"}==pred_input['args']['operation'], 'Prediction failed!'