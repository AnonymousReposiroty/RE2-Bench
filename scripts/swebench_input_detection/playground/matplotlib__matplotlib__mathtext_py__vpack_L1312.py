# Problem: matplotlib@@matplotlib__mathtext.py@@vpack_L1312
# Module: matplotlib._mathtext
# Function: vpack
# Line: 1312

from matplotlib._mathtext import Vlist, Hlist, Vbox, Hbox, Glue, Kern, List
import numpy as np


def create_children():
    """
    Create the children based on the JSON representation:
    1. HCentered<w=6.11 h=1.83 d=0.00 s=0.00>[Glue, Hbox, `\combiningdotabove`, Glue]
    2. Vbox
    3. Hlist<w=7.80 h=7.38 d=0.19 s=0.00>[`s`, k1.69]
    """
    # Child 1: HCentered - create as Hlist with proper dimensions
    # HCentered is an Hlist subclass, we create equivalent Hlist
    child1 = List.__new__(Hlist)
    List.__init__(child1, [])
    child1.width = 6.11
    child1.height = 1.83
    child1.depth = 0.0
    child1.shift_amount = 0.0
    child1.glue_set = 0.0
    child1.glue_sign = 0
    child1.glue_order = 0

    # Child 2: Vbox with default values
    child2 = Vbox(0.0, 0.0)

    # Child 3: Hlist with specific dimensions
    child3 = List.__new__(Hlist)
    List.__init__(child3, [])
    child3.width = 7.80
    child3.height = 7.38
    child3.depth = 0.19
    child3.shift_amount = 0.0
    child3.glue_set = 0.0
    child3.glue_sign = 0
    child3.glue_order = 0

    return [child1, child2, child3]


def test_input(pred_input):
    # Create ground truth Vlist with properly initialized children
    children_gt = create_children()

    obj_ins = List.__new__(Vlist)
    List.__init__(obj_ins, children_gt)
    obj_ins.size = 0
    obj_ins.width = 0.0
    obj_ins.height = 0.0
    obj_ins.depth = 0.0
    obj_ins.shift_amount = 0.0
    obj_ins.glue_set = 0.0
    obj_ins.glue_sign = 0
    obj_ins.glue_order = 0

    # Create predicted Vlist with properly initialized children
    children_pred = create_children()

    obj_ins_pred = List.__new__(Vlist)
    List.__init__(obj_ins_pred, children_pred)
    obj_ins_pred.size = pred_input['self']['size']
    obj_ins_pred.width = pred_input['self']['width']
    obj_ins_pred.height = pred_input['self']['height']
    obj_ins_pred.depth = pred_input['self']['depth']
    obj_ins_pred.shift_amount = pred_input['self']['shift_amount']
    obj_ins_pred.glue_set = pred_input['self']['glue_set']
    obj_ins_pred.glue_sign = pred_input['self']['glue_sign']
    obj_ins_pred.glue_order = pred_input['self']['glue_order']

    # Call vpack on both and compare results
    result_gt = obj_ins.vpack(h=0.0, m='additional', l=np.inf)
    result_pred = obj_ins_pred.vpack(h=pred_input['args']['h'], m=pred_input['args']['m'], l=pred_input['args']['l'])

    assert result_gt == result_pred, 'Prediction failed!'


