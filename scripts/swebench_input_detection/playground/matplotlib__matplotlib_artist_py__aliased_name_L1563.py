# Problem: matplotlib@@matplotlib_artist.py@@aliased_name_L1563
# Module: matplotlib.artist
# Function: aliased_name
# Line: 1563

from matplotlib.artist import ArtistInspector


def test_input(pred_input):
    obj_ins = ArtistInspector(o = "<class 'matplotlib.tests.test_artist.test_artist_inspector_get_valid_values.<locals>.TestArtist'>")
    obj_ins.oorig = "<class 'matplotlib.tests.test_artist.test_artist_inspector_get_valid_values.<locals>.TestArtist'>"
    obj_ins.aliasd = {}
    obj_ins_pred = ArtistInspector(o = pred_input['self']['o'])
    obj_ins_pred.oorig = pred_input['self']['oorig']
    obj_ins_pred.aliasd = pred_input['self']['aliasd']
    assert obj_ins.aliased_name(s = 'clip_on')==obj_ins_pred.aliased_name(s = pred_input['args']['s']), 'Prediction failed!'
