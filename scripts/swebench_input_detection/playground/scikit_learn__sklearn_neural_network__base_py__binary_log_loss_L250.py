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

# Problem: scikit-learn@@sklearn_neural_network__base.py@@binary_log_loss_L250
# Module: sklearn.neural.network._base
# Function: binary_log_loss
# Line: 250

from sklearn.neural_network._base import binary_log_loss


def test_input(pred_input):
    assert binary_log_loss(y_true = np.array([[ True, False, True], [ True, True, True], [ True, False, True], [False, True, True], [False, False, False], [False, True, False], [False, True, False], [ True, False, True], [ True, False, True], [False, True, False], [ True, True, True], [ True, True, True], [ True, True, True], [False, True, False], [False, False, False], [ True, True, True], [ True, True, True], [False, True, True], [ True, True, False], [False, True, False], [ True, True, True], [False, True, False], [ True, True, False], [ True, True, False], [ True, True, True], [ True, True, True], [False, False, False], [False, True, True], [ True, False, True], [False, False, False], [False, True, True], [False, True, False], [False, True, True], [ True, True, True], [False, True, False], [False, True, True], [ True, True, False], [False, False, False], [False, False, False], [False, True, False], [ True, True, True], [False, False, True], [ True, True, False], [ True, True, False], [False, True, True], [ True, True, True], [False, True, True], [ True, False, False], [False, False, True], [ True, True, True], [ True, True, False], [ True, True, True], [False, False, False], [False, False, False], [False, True, True], [ True, True, True], [False, True, True], [ True, True, True], [False, True, True], [False, True, False], [False, True, True], [ True, True, False], [False, False, False], [False, False, True], [False, True, True], [False, False, False], [False, False, False], [False, False, True], [False, True, True], [ True, False, True], [ True, True, False], [ True, False, True], [False, True, True], [False, True, True], [False, True, True]]), y_prob = np.array([[0.22321879, 0.85032391, 0.08316667], [0.42508744, 0.7303509 , 0.02641982], [0.07765612, 0.74986681, 0.15777179], [0.05942839, 0.81653183, 0.02691593], [0.34594303, 0.84595066, 0.05353217], [0.16317711, 0.57759888, 0.02615454], [0.2177721 , 0.59022332, 0.03371363], [0.2259984 , 0.35858696, 0.00385401], [0.15718466, 0.81171412, 0.03203554], [0.17491617, 0.45803566, 0.040298 ], [0.23214869, 0.80999503, 0.13816021], [0.14016061, 0.81278042, 0.08162685], [0.55404741, 0.93233064, 0.05532857], [0.17221141, 0.50406182, 0.03574973], [0.34593664, 0.88691045, 0.06022066], [0.43207108, 0.85709736, 0.07615062], [0.38772095, 0.6102242 , 0.19851966], [0.5066754 , 0.79425977, 0.01559495], [0.31204832, 0.60981564, 0.04216514], [0.09764651, 0.12544553, 0.00570331], [0.18102261, 0.83308026, 0.30874512], [0.27573005, 0.73000604, 0.02723533], [0.33027774, 0.81956165, 0.17379279], [0.1874203 , 0.56088916, 0.10514311], [0.27908401, 0.38597783, 0.02204938], [0.09106593, 0.94988169, 0.02600111], [0.12689932, 0.92850399, 0.17791372], [0.16085829, 0.8658684 , 0.09061511], [0.17623172, 0.76284335, 0.0075743 ], [0.38682134, 0.93121668, 0.33341611], [0.20472078, 0.68375116, 0.02197602], [0.22199996, 0.64899547, 0.03817049], [0.15752389, 0.90500031, 0.04282205], [0.20195947, 0.6855218 , 0.04511383], [0.15089524, 0.40386241, 0.02018433], [0.06586679, 0.87610654, 0.08187604], [0.52525707, 0.87031821, 0.08370813], [0.39229069, 0.96267987, 0.06380436], [0.23473095, 0.5626479 , 0.12531144], [0.16365109, 0.63693444, 0.02512276], [0.35053448, 0.86557225, 0.21212595], [0.14770935, 0.82690444, 0.02516767], [0.30530367, 0.79132758, 0.03692632], [0.16688682, 0.56945568, 0.04062158], [0.3534248 , 0.79881565, 0.11170072], [0.09541751, 0.83226547, 0.06868634], [0.14541013, 0.78334874, 0.05996865], [0.65016001, 0.75455587, 0.04494548], [0.04840142, 0.54446019, 0.01148331], [0.33315448, 0.5876401 , 0.05768731], [0.20153076, 0.83307631, 0.1113276 ], [0.17730159, 0.94791689, 0.05209839], [0.26609141, 0.72192386, 0.11297701], [0.4227562 , 0.78974921, 0.32919064], [0.13732484, 0.61828099, 0.0389986 ], [0.30620817, 0.82414999, 0.01841289], [0.27919241, 0.64358832, 0.07005714], [0.28463113, 0.43143134, 0.04949251], [0.10044238, 0.5857955 , 0.12373204], [0.23221559, 0.2951493 , 0.03930493], [0.09245144, 0.52652324, 0.05007232], [0.406501 , 0.74804229, 0.3079554 ], [0.15635617, 0.87904465, 0.1267242 ], [0.12963608, 0.8103028 , 0.07898413], [0.2053942 , 0.7561511 , 0.05825872], [0.42034923, 0.95265089, 0.14819679], [0.38109966, 0.82716352, 0.113135 ], [0.09190947, 0.41013448, 0.00938115], [0.05923477, 0.41740088, 0.02987121], [0.24598814, 0.79940836, 0.12921287], [0.1048954 , 0.71421076, 0.02349611], [0.22208945, 0.96208365, 0.16351097], [0.14449632, 0.65588259, 0.02195946], [0.16371734, 0.77336183, 0.05020408], [0.19101809, 0.70658047, 0.05818781]]), sample_weight = None)==binary_log_loss(y_true = pred_input['args']['y_true'], y_prob = pred_input['args']['y_prob'], sample_weight = pred_input['args']['sample_weight']), 'Prediction failed!'