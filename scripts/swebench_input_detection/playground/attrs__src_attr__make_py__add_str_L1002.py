# Problem: attrs@@src_attr__make.py@@add_str_L1002
# Module: attr._make
# Function: add_str
# Line: 1002

from attr._make import _ClassBuilder
from attrs import ClassProps
class A:
    x = 1
    y = 2
    def __repr__(self):
        return f"A(x={self.x}, y={self.y})"
    
props = ClassProps(
    is_exception=False,
    is_slotted=False,
    has_weakref_slot=False,
    is_frozen=False,
    kw_only=ClassProps.KeywordOnly.NO,
    collected_fields_by_mro=True,
    added_init=True,                        
    added_repr=True,
    added_eq=True,
    added_ordering=False,
    hashability=ClassProps.Hashability.LEAVE_ALONE,
    added_match_args=True,
    added_str=False,
    added_pickling=True,
    on_setattr_hook=None,
    field_transformer=None,
)


def test_input(pred_input):
    obj_ins = _ClassBuilder(cls=A, auto_attribs = False, these = None, has_custom_setattr = False, props=props)
    obj_ins_pred = _ClassBuilder(cls=A, auto_attribs = pred_input['self']['auto_attribs'], these = pred_input['self']['these'], has_custom_setattr = pred_input['self']['has_custom_setattr'], props=props)
    assert obj_ins.add_str()==obj_ins_pred.add_str(), 'Prediction failed!'