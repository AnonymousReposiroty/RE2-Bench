# Problem: matplotlib@@matplotlib_artist.py@@__init_subclass___L119
# Module: matplotlib.artist
# Function: __init_subclass__
# Line: 119

from matplotlib.artist import Artist


def test_input(pred_input):
    obj_ins = Artist()
    obj_ins_pred = Artist()
    assert obj_ins.__init_subclass__(cls = "<class 'matplotlib.tests.test_artist.test_artist_inspector_get_valid_values.<locals>.TestArtist'>")==obj_ins_pred.__init_subclass__(cls = pred_input['args']['cls']), 'Prediction failed!'