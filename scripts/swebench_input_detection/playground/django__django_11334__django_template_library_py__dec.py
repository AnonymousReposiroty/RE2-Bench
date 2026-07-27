# Problem: django__django-11334@@django.template.library.py@@dec
# Benchmark: Swebench
# Module: django.template.library
# Function: dec

from django.template.library import Library


def test_input(pred_input):
    obj_ins = Library()
    obj_ins_pred = Library()
    assert {'__module__': 'template_tests.test_library', '__name__': 'func', '__qualname__': 'SimpleTagRegistrationTests.test_simple_tag_wrapped."<locals>".func', '__doc__': None, '__annotations__': {}, '__wrapped__': {}} == pred_input['args']['func'], 'Prediction failed!'
    