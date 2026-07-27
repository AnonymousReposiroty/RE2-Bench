# Problem: sympy__sympy-22969@@sympy.physics.optics.gaussopt.py@@waist2rayleigh
# Benchmark: Swebench
# Module: sympy.physics.optics.gaussopt
# Function: waist2rayleigh

from sympy.physics.optics.gaussopt import waist2rayleigh
from sympy import sympify

def test_input(pred_input):
    assert waist2rayleigh(w = sympify('w_0'), wavelen = sympify('l'), n = sympify('1'))==waist2rayleigh(w = sympify(pred_input['args']['w']), wavelen = sympify(pred_input['args']['wavelen']), n = sympify(pred_input['args']['n'])), 'Prediction failed!'
    