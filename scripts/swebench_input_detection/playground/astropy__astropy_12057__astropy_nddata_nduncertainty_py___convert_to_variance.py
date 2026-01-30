# Problem: astropy__astropy-12057@@astropy.nddata.nduncertainty.py@@_convert_to_variance
# Benchmark: Swebench
# Module: astropy.nddata.nduncertainty
# Function: _convert_to_variance

from astropy.nddata.nduncertainty import VarianceUncertainty
import numpy as np

def test_input(pred_input):
    obj_ins = VarianceUncertainty()
    obj_ins._unit = {'_bases': None, '_powers': None, '_scale': 1.0}
    obj_ins._array = '[[  1.   4.   9.  16.  25.]\n [ 36.  49.  64.  81. 100.]]'
    obj_ins._parent_nddata = None
    obj_ins_pred = VarianceUncertainty()
    obj_ins_pred._unit = pred_input['self']['_unit']
    obj_ins_pred._array = pred_input['self']['_array']
    obj_ins_pred._parent_nddata = pred_input['self']['_parent_nddata']
    assert obj_ins._convert_to_variance()==obj_ins_pred._convert_to_variance(), 'Prediction failed!'
    