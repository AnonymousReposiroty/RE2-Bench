from sklearn.utils import Bunch, check_array, check_random_state

def _shuffle(data, random_state=None):
    generator = check_random_state(random_state)
    n_rows, n_cols = data.shape
    row_idx = generator.permutation(n_rows)
    col_idx = generator.permutation(n_cols)
    result = data[row_idx][:, col_idx]
    return result, row_idx, col_idx
