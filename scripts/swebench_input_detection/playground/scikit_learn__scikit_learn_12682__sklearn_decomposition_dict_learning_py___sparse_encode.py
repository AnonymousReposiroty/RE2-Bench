import numpy as np
from io import StringIO
def string2Array(s):
    s = s.strip()

    # String case (elements quoted, no commas, arbitrary newlines)
    if "'" in s or '"' in s:
        # Remove brackets & quotes, collapse all whitespace to single spaces,
        # then parse as a single whitespace-delimited row of strings.
        payload = (
            s.replace('[', ' ').replace(']', ' ')
             .replace("'", ' ').replace('"', ' ')
        )
        payload = ' '.join(payload.split())
        arr = np.loadtxt(StringIO(payload), dtype=str, comments=None)
        return np.atleast_1d(arr)

    # Numeric case (works for 1D or 2D pretty-printed arrays)
    txt = s.replace('[', '').replace(']', '')
    try:
        return np.loadtxt(StringIO(txt), dtype=float, comments=None)
    except ValueError:
        # If wrapping caused uneven rows, flatten newlines to a single row
        txt_one_line = ' '.join(txt.split())
        return np.loadtxt(StringIO(txt_one_line), dtype=float, comments=None)


from sklearn.decomposition import sparse_encode

def test_input(pred_input):
    X=np.array([[ 0.17742614,-0.40178094,-1.63019835,0.46278226,-0.90729836,0.0519454,0.72909056,0.12898291],
        [ 0.06651722,0.3024719,-0.63432209,-0.36274117,-0.67246045,-0.35955316,-0.81314628,-1.7262826],
        [-0.10321885, 0.4105985,  0.14404357, 1.45427351, 0.76103773, 0.12167502, 0.44386323, 0.33367433]])
    dictionary = np.array([[0.46135836,  0.10465442,  0.25597254,  0.5860681,   0.48842853, -0.25559067, 0.24847972, -0.03958495],
        [0.52852955, -0.57279405,  0.18663272, -0.31766031, -0.40393343, -0.26850884, -0.14451871,  0.02605321],
        [-0.37626088, -0.36852725, -0.49989885,  0.5125477,  -0.17089141, -0.19495868, -0.3042246,   0.22415631],
        [ 0.65578101, -0.42019757,  0.01322062, -0.05408145,  0.44285294,  0.4245294, 0.04476765,  0.1092593],  
        [ 0.81995322, -0.44790005,  0.03641964,  0.2705602, -0.10797034,  0.09925906,  0.17593026, -0.00846372],
        [ 0.44358159, -0.06091005, 0.0929476, -0.25357503, -0.75796475,  0.19405477,  0.25664504, -0.22034358],
        [-0.31274795, -0.6977948, -0.12256234,  0.05507848,  0.43340694,  0.4235745, -0.13644729, -0.10649495],
        [-0.05754613, 0.22891565, 0.0803068, 0.81078229,  0.42429157 , 0.06783581,  0.24746134,  0.18602908]])

    gram = np.array([[ 1,        -0.12010716, -0.15783711,  0.34485831,  0.46525134,  -0.27386145,  -0.14267121,  0.7371607 ],
        [-0.12010716,  1,         -0.0727053,   0.31043704,  0.60209003,  0.57846389,  -0.0778306,  -0.62461803],
        [-0.15783711, -0.0727053,   1,         -0.27379249, -0.07930379, -0.35666122,   0.3253244,   0.19339282],
        [ 0.34485831,  0.31043704, -0.27379249,  1,          0.71304012,  0.06555902,   0.43752982,  0.0713867 ],
        [ 0.46525134,  0.60209003, -0.07930379,  0.71304012,  1,          0.47389153,   0.03868666 , 0.07545756],
        [-0.27386145,  0.57846389, -0.35666122,  0.06555902,  0.47389153,  1,  -0.37944849, -0.52351426],
        [-0.14267121, -0.0778306,   0.3253244,   0.43752982,  0.03868666, -0.37944849,   1,       0.05212319],
        [ 0.7371607,  -0.62461803,  0.19339282,  0.0713867,   0.07545756, -0.52351426,   0.05212319,  1   ]])

    assert sparse_encode(X=X, dictionary=dictionary, gram=gram, cov=None, algorithm='lasso_cd', copy_cov=True, init=None, max_iter=1000, check_input=False, verbose=False) == sparse_encode(X = pred_input['args']['X'], dictionary = pred_input['args']['dictionary'], gram = pred_input['args']['gram'], cov = pred_input['kwargs']['cov'], algorithm = pred_input['kwargs']['algorithm'], copy_cov = pred_input['kwargs']['copy_cov'], init = pred_input['kwargs']['init'], max_iter = pred_input['kwargs']['max_iter'], check_input = pred_input['kwargs']['check_input'], verbose = pred_input['kwargs']['verbose']), 'Prediction failed!'
 


