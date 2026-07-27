import numpy as np
from sklearn.utils.random import sample_without_replacement

def _generate_hypercube(samples, dimensions, rng):
    """Returns distinct binary samples of length dimensions."""
    if dimensions > 30:
        return np.hstack(
            [
                rng.randint(2, size=(samples, dimensions - 30)),
                _generate_hypercube(samples, 30, rng),
            ]
        )
    out = sample_without_replacement(2**dimensions, samples, random_state=rng).astype(
        dtype=">u4", copy=False
    )
    out = np.unpackbits(out.view(">u1")).reshape((-1, 32))[:, -dimensions:]
    return out
