# Problem: django__django-11808@@django.db.models.base.py@@__eq__
# Benchmark: Swebench
# Module: django.db.models.base
# Function: __eq__
import django
from django.conf import settings

# Minimal settings for a standalone script
settings.configure(
    INSTALLED_APPS=["django.contrib.contenttypes"],
    DATABASES={"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}},
)
django.setup()

from django.db import models

class Entry(models.Model):
    headline = models.CharField(max_length=100)
    pub_date = models.DateTimeField()

    class Meta:
        app_label = "tmpapp"   # required since we’re not in a real Django app
        managed = False   

def test_input(pred_input):
    obj_ins = Entry(id=1, headline="First", pub_date="2014-05-16 12:01:00")
    obj_ins_pred = Entry(id=pred_input['self']['id'], headline=pred_input['self']['headline'], pub_date=pred_input['self']['pub_date'])
    other = Entry(id=1, headline="First", pub_date='2014-05-16 12:01:00')
    assert obj_ins.__eq__(other = other)==obj_ins_pred.__eq__(other = other), 'Prediction failed!'
    
    