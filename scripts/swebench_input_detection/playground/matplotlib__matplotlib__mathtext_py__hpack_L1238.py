# Problem: matplotlib@@matplotlib__mathtext.py@@hpack_L1238
# Module: matplotlib._mathtext
# Function: hpack
# Line: 1238

from matplotlib._mathtext import Hlist


def test_input(pred_input):
    obj_ins = Hlist(elements=[])
    obj_ins.size = 0
    obj_ins.width = 0.0
    obj_ins.height = 0.0
    obj_ins.depth = 0.0
    obj_ins.shift_amount = 0.0
    obj_ins.children = ['Hlist<w=10.58 h=13.00 d=0.00 s=0.00>[\n  `0`,\n]', 'Hlist<w=8.08 h=9.10 d=0.00 s=-7.00>[\n  k0.17,\n  Hlist<w=7.40 h=9.10 d=0.00 s=0.00>[\n    `0`,\n    k1.54,\n  ],\n]']
    obj_ins.glue_set = 0.0
    obj_ins.glue_sign = 0
    obj_ins.glue_order = 0
    obj_ins_pred = Hlist(elements=[])
    obj_ins_pred.size = pred_input['self']['size']
    obj_ins_pred.width = pred_input['self']['width']
    obj_ins_pred.height = pred_input['self']['height']
    obj_ins_pred.depth = pred_input['self']['depth']
    obj_ins_pred.shift_amount = pred_input['self']['shift_amount']
    obj_ins_pred.children = pred_input['self']['children']
    obj_ins_pred.glue_set = pred_input['self']['glue_set']
    obj_ins_pred.glue_sign = pred_input['self']['glue_sign']
    obj_ins_pred.glue_order = pred_input['self']['glue_order']
    assert obj_ins.hpack(w = 0.0, m = 'additional')==obj_ins_pred.hpack(w = pred_input['args']['w'], m = pred_input['args']['m']), 'Prediction failed!'
    
    
    
