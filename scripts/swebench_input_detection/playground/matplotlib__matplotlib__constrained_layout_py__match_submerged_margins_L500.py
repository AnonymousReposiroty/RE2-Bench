# Problem: matplotlib@@matplotlib__constrained_layout.py@@match_submerged_margins_L500
# Module: matplotlib._constrained.layout
# Function: match_submerged_margins
# Line: 500

from  matplotlib._constrained_layout import match_submerged_margins



def test_input(pred_input):
    assert match_submerged_margins(layoutgrids = {'hasgrids': True, 'Figure(1000x500)': 'LayoutBox: figlb000002               1x1,\n0, 0: L0.000, B0.000, R1.000, T1.000, ML0.000, MR0.000, MB0.000, MT0.000, \n', 'GridSpec(1, 2)': 'LayoutBox: figlb000002.gridspec000003 1x2,\n0, 0: L0.000, B0.000, R0.486, T1.000, ML0.027, MR0.000, MB0.047, MT0.047, \n0, 1: L0.486, B0.000, R1.000, T1.000, ML0.027, MR0.027, MB0.047, MT0.047, \n'}, fig = '<Figure size 1000x500 with 2 Axes>')==match_submerged_margins(layoutgrids = pred_input['args']['layoutgrids'], fig = pred_input['args']['fig']), 'Prediction failed!'