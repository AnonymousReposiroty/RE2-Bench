# Problem: packaging@@src_packaging_tags.py@@_mac_arch_L357
# Module: packaging.tags
# Function: _mac_arch
# Line: 357

from packaging.tags import _mac_arch


def test_input(pred_input):
    assert _mac_arch(arch = 'x86_64', is_32bit = True)==_mac_arch(arch = pred_input['args']['arch'], is_32bit = pred_input['args']['is_32bit']), 'Prediction failed!'
    
    
_mac_arch(arch = 'x86_64', is_32bit = True)