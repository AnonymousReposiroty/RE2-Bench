import numpy as np
from io import StringIO
def string2Array(s):
    return np.loadtxt(StringIO(s.replace('[','').replace(']','')))

from sklearn.datasets import make_blobs

def test_input(pred_input):
	assert make_blobs(n_samples = 1, centers = [[0,0]], random_state = 0)==make_blobs(n_samples = pred_input['kwargs']['n_samples'], centers = pred_input['kwargs']['centers'], random_state = pred_input['kwargs']['random_state']), 'Prediction failed!'
 
